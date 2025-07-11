#!/usr/bin/env python3
"""
Smart warehouse training with a STABILIZED Proximal Policy Optimization (PPO) agent.
This script includes several improvements to stabilize learning and improve performance,
while maintaining full compatibility with the warehouse_visualization.py script.
"""
import os
import time
import json
import pickle
import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import mlflow
from datetime import datetime
from collections import deque

# Set proxy bypass for MLflow connection
os.environ['NO_PROXY'] = '127.0.0.1,localhost'
os.environ['no_proxy'] = '127.0.0.1,localhost'


# ################################## PPO Agent and Network ###################################

class Memory:
    """A buffer for storing trajectories experienced by a PPO agent."""
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    """An Actor-Critic network with shared layers for the PPO agent."""
    def __init__(self, state_dim, action_dim, action_std_init):
        super(ActorCritic, self).__init__()
        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init)

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std)

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action_mean = self.actor(state)
        # Create a distribution with standard deviation, not a covariance matrix
        action_std = torch.sqrt(self.action_var)
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)
        action_var = self.action_var.expand_as(action_mean)
        action_std = torch.sqrt(action_var)
        dist = Normal(action_mean, action_std)
        action_logprobs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        state_values = self.critic(state)
        return state_values, action_logprobs, dist_entropy


class PPO:
    """The Stabilized PPO agent."""
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std_init=0.6):
        self.action_std = action_std_init
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, action_std_init)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, action_std_init)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.action_std = max(self.action_std - action_std_decay_rate, min_action_std)
        self.set_action_std(self.action_std)

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state)
            action, action_logprob = self.policy_old.act(state)
        return action.numpy().flatten(), action_logprob.sum().numpy()

    def update(self, memory):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalize rewards
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Convert lists to tensors
        old_states = torch.squeeze(torch.stack(memory.states, dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions, dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            state_values, logprobs, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            self.optimizer.zero_grad()
            loss.mean().backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())


# ################################## Warehouse Environment ###################################

class SmartWarehouseEnv:
    """Enhanced warehouse environment with smart rewards and state representation."""
    def __init__(self):
        self.state_dim = 23
        self.action_dim = 2
        self.max_steps = 800
        self.current_step = 0
        self.agent_pos = np.array([0.0, 0.0])
        self.agent_angle = 0.0
        self.agent_vel = np.array([0.0, 0.0])
        self.package_pos = np.array([0.0, 2.0])
        self.delivery_pos = np.array([5.0, 5.0])
        self.has_package = False
        self.obstacles = [
            np.array([3.0, 2.0]), np.array([-3.0, -2.0]),
            np.array([1.0, 4.0]), np.array([-1.0, -4.0])
        ]
        self.position_history = []
        self.episode_data = []
        self.last_distance_to_package = None
        self.last_distance_to_delivery = None
        self.delivered_successfully = False

    def reset(self):
        self.current_step = 0
        self.agent_pos = np.array([0.0, 0.0])
        self.agent_angle = np.random.uniform(0, 2 * np.pi)
        self.agent_vel = np.array([0.0, 0.0])
        self.package_pos = np.array([np.random.uniform(-2, 2), np.random.uniform(1, 4)])
        self.has_package = False
        self.position_history = []
        self.delivered_successfully = False
        self.last_distance_to_package = np.linalg.norm(self.agent_pos - self.package_pos)
        self.last_distance_to_delivery = np.linalg.norm(self.agent_pos - self.delivery_pos)
        return self._get_observation()

    def _get_observation(self):
        obs = [
            self.agent_pos[0], self.agent_pos[1], np.cos(self.agent_angle), np.sin(self.agent_angle),
            self.agent_vel[0], self.agent_vel[1], self.package_pos[0],
            self.package_pos[1], float(self.has_package), self.delivery_pos[0],
            self.delivery_pos[1]
        ]
        if not self.has_package:
            dist_to_package = np.linalg.norm(self.agent_pos - self.package_pos)
            direction = (self.package_pos - self.agent_pos) / (dist_to_package + 1e-5)
            obs.extend([dist_to_package, 0.0])
        else:
            dist_to_delivery = np.linalg.norm(self.agent_pos - self.delivery_pos)
            direction = (self.delivery_pos - self.agent_pos) / (dist_to_delivery + 1e-5)
            obs.extend([0.0, dist_to_delivery])
        obs.extend([direction[0], direction[1]])
        directions = np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]])
        for dir_vec in directions:
            min_dist = 10.0
            for obs_pos in self.obstacles:
                to_obstacle = obs_pos - self.agent_pos
                proj = np.dot(to_obstacle, dir_vec)
                if proj > 0:
                    cross_dist = np.linalg.norm(to_obstacle - proj * dir_vec)
                    if cross_dist < 0.5:
                        min_dist = min(min_dist, proj)
            obs.append(min(min_dist / 10.0, 1.0))
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        left_vel, right_vel = np.clip(action, -1, 1)
        v = (left_vel + right_vel) / 2.0
        omega = (right_vel - left_vel) / 0.5
        dt = 0.1
        self.agent_angle += omega * dt
        self.agent_pos[0] += v * np.cos(self.agent_angle) * dt
        self.agent_pos[1] += v * np.sin(self.agent_angle) * dt
        self.agent_vel = np.array([v, omega])
        self.position_history.append({
            'step': self.current_step, 'agent_pos': self.agent_pos.copy(),
            'agent_angle': self.agent_angle, 'package_pos': self.package_pos.copy(),
            'has_package': self.has_package, 'timestamp': time.time()
        })
        reward, done = self._calculate_smart_reward()
        return self._get_observation(), reward, done, {}

    def _calculate_smart_reward(self):
        reward, done = -0.05, False
        if abs(self.agent_vel[1]) > 0.8 and abs(self.agent_vel[0]) < 0.1: reward -= 0.1
        if not self.has_package:
            dist_to_package = np.linalg.norm(self.agent_pos - self.package_pos)
            reward += (self.last_distance_to_package - dist_to_package) * 10
            self.last_distance_to_package = dist_to_package
            if dist_to_package < 0.3:
                self.has_package = True
                reward += 100
                self.last_distance_to_delivery = np.linalg.norm(self.agent_pos - self.delivery_pos)
        else:
            dist_to_delivery = np.linalg.norm(self.agent_pos - self.delivery_pos)
            reward += (self.last_distance_to_delivery - dist_to_delivery) * 50
            self.last_distance_to_delivery = dist_to_delivery
            if dist_to_delivery < 0.5:
                reward += 1000 + 100 * (self.max_steps - self.current_step) / self.max_steps
                self.delivered_successfully = True
                done = True
        for obs in self.obstacles:
            if np.linalg.norm(self.agent_pos - obs) < 0.6:
                reward -= 50
                done = True
                break
        if self.current_step >= self.max_steps:
            done = True
            reward -= 100
        return reward, done

    def save_episode_data(self, episode_num, reward, success):
        episode_info = {
            'episode': episode_num, 'positions': self.position_history.copy(),
            'final_reward': reward, 'success': success,
            'obstacles': [o.tolist() for o in self.obstacles],
            'delivery_pos': self.delivery_pos.tolist()
        }
        self.episode_data.append(episode_info)
        os.makedirs("training_data", exist_ok=True)
        with open(f"training_data/episode_{episode_num}.pkl", "wb") as f: pickle.dump(episode_info, f)
        with open("training_data/latest_episode.pkl", "wb") as f: pickle.dump(episode_info, f)
        success_rate = sum(ep['success'] for ep in self.episode_data) / len(self.episode_data) if self.episode_data else 0.0
        progress = {
            'current_episode': episode_num, 'total_episodes': len(self.episode_data),
            'latest_reward': reward, 'success_rate': success_rate
        }
        with open("training_data/progress.json", "w") as f: json.dump(progress, f)


# ################################## Main Training Function ###################################

def train_ppo_agent():
    """Main training function with stabilized PPO and adaptive exploration."""
    print("🧠 Starting Warehouse Agent Training with STABILIZED PPO & ADAPTIVE EXPLORATION...")
    print("=" * 60)

    # --- Hyperparameters ---
    env = SmartWarehouseEnv()
    state_dim = env.state_dim
    action_dim = env.action_dim
    total_training_timesteps = int(2e6)
    action_std_init = 0.6
    action_std_decay_rate = 0.005
    min_action_std = 0.1
    update_timestep = 2048
    K_epochs = 40
    eps_clip = 0.2
    gamma = 0.99
    lr_actor = 3e-4
    lr_critic = 1e-3

    # --- Initialization ---
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std_init)
    memory = Memory()

    # --- Adaptive Exploration ---
    stagnation_window = 100
    success_history = deque(maxlen=stagnation_window)
    best_success_rate = 0.0
    stagnation_counter = 0
    
    start_time = datetime.now().replace(microsecond=0)
    time_step, i_episode = 0, 0
    
    # --- Training Loop ---
    while time_step <= total_training_timesteps:
        i_episode += 1
        state = env.reset()
        current_ep_reward = 0
        
        for t in range(1, env.max_steps + 1):
            action, log_prob = ppo_agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            memory.states.append(torch.FloatTensor(state))
            memory.actions.append(torch.from_numpy(action))
            memory.logprobs.append(torch.tensor(log_prob))
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            if len(memory.states) >= update_timestep:
                ppo_agent.update(memory)
                memory.clear_memory()

            if time_step % 2000 == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            state = next_state
            if done: break
        
        success_history.append(1 if env.delivered_successfully else 0)
        
        # --- Adaptive Exploration Check ---
        if i_episode > 0 and i_episode % stagnation_window == 0:
            current_success_rate = np.mean(success_history)
            print(f"\n📈 Checking for stagnation: Current success rate ({current_success_rate:.2%}) vs Best ({best_success_rate:.2%})")
            if current_success_rate <= best_success_rate + 0.01:
                stagnation_counter += 1
                if stagnation_counter >= 2:
                    old_std = ppo_agent.action_std
                    new_std = min(old_std + 0.1, action_std_init)
                    ppo_agent.set_action_std(new_std)
                    print(f"📉 Performance stagnated. Boosting exploration from {old_std:.2f} to {new_std:.2f}!")
                    stagnation_counter = 0
            else:
                best_success_rate = current_success_rate
                stagnation_counter = 0

        env.save_episode_data(i_episode, current_ep_reward, env.delivered_successfully)
        
        if i_episode % 20 == 0:
            avg_success = np.mean(list(success_history))
            print(f"Epi: {i_episode:4d} | Timestep: {time_step:7d} | Reward: {current_ep_reward:8.2f} | Avg Success (last {len(success_history)}): {avg_success:.2%} | Action Std: {ppo_agent.action_std:.2f}")

    print("\n🎉 Training finished!")
    print(f"Total time taken: {datetime.now().replace(microsecond=0) - start_time}")

if __name__ == "__main__":
    train_ppo_agent()

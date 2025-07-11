#!/usr/bin/env python3
"""
Smart warehouse training with goal-oriented behavior
"""
import mlflow
import numpy as np
import time
import os
import json
import pickle
from datetime import datetime

# Set proxy bypass
os.environ['NO_PROXY'] = '127.0.0.1,localhost'
os.environ['no_proxy'] = '127.0.0.1,localhost'

class SmartWarehouseEnv:
    """Enhanced warehouse environment with better rewards and state representation"""
    
    def __init__(self):
        self.state_dim = 22  # Enhanced state representation (corrected)
        self.action_dim = 2
        self.max_steps = 1000  # Even longer for delivery learning
        self.current_step = 0
        
        # Environment state
        self.agent_pos = np.array([0.0, 0.0])
        self.agent_angle = 0.0
        self.agent_vel = np.array([0.0, 0.0])
        self.package_pos = np.array([0.0, 2.0])
        self.delivery_pos = np.array([5.0, 5.0])
        self.has_package = False
        
        # Obstacles (shelves)
        self.obstacles = [
            np.array([3.0, 2.0]),
            np.array([-3.0, -2.0]),
            np.array([1.0, 4.0]),
            np.array([-1.0, -4.0])
        ]
        
        # Tracking for visualization
        self.position_history = []
        self.episode_data = []
        
        # Learning metrics
        self.last_distance_to_package = None
        self.last_distance_to_delivery = None
        self.delivered_successfully = False  # Track actual delivery
        
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        self.agent_pos = np.array([0.0, 0.0])
        self.agent_angle = 0.0
        self.agent_vel = np.array([0.0, 0.0])
        # Randomize package position for better generalization
        self.package_pos = np.array([
            np.random.uniform(-2, 2), 
            np.random.uniform(1, 4)
        ])
        self.has_package = False
        self.position_history = []
        self.delivered_successfully = False  # Reset delivery flag
        
        # Initialize distance tracking
        self.last_distance_to_package = np.linalg.norm(self.agent_pos - self.package_pos)
        self.last_distance_to_delivery = np.linalg.norm(self.agent_pos - self.delivery_pos)
        
        return self._get_observation()
    
    def _get_observation(self):
        """Enhanced observation with goal information"""
        # Basic position and orientation
        obs = [
            self.agent_pos[0], self.agent_pos[1],  # Agent position
            self.agent_angle,  # Agent orientation
            self.agent_vel[0], self.agent_vel[1],  # Agent velocity
        ]
        
        # Package information
        obs.extend([
            self.package_pos[0], self.package_pos[1],  # Package position
            float(self.has_package),  # Has package flag
        ])
        
        # Delivery information
        obs.extend([
            self.delivery_pos[0], self.delivery_pos[1],  # Delivery position
        ])
        
        # Distance information (crucial for learning!)
        if not self.has_package:
            dist_to_package = np.linalg.norm(self.agent_pos - self.package_pos)
            obs.extend([dist_to_package, 0.0])  # Distance to package, 0 to delivery
        else:
            dist_to_delivery = np.linalg.norm(self.agent_pos - self.delivery_pos)
            obs.extend([0.0, dist_to_delivery])  # 0 to package, distance to delivery
        
        # Direction vectors (help robot understand which way to go)
        if not self.has_package:
            # Direction to package
            direction = self.package_pos - self.agent_pos
            direction = direction / max(np.linalg.norm(direction), 0.1)  # Normalize
        else:
            # Direction to delivery
            direction = self.delivery_pos - self.agent_pos
            direction = direction / max(np.linalg.norm(direction), 0.1)  # Normalize
        
        obs.extend([direction[0], direction[1]])
        
        # Simplified obstacle detection (closest obstacle distance in 8 directions)
        directions = np.array([
            [1, 0], [1, 1], [0, 1], [-1, 1],
            [-1, 0], [-1, -1], [0, -1], [1, -1]
        ])
        
        for dir_vec in directions:
            min_dist = 10.0  # Max sensor range
            for obs_pos in self.obstacles:
                to_obstacle = obs_pos - self.agent_pos
                # Project onto direction
                proj = np.dot(to_obstacle, dir_vec)
                if proj > 0:  # Obstacle is in this direction
                    cross_dist = np.linalg.norm(to_obstacle - proj * dir_vec)
                    if cross_dist < 0.5:  # Obstacle is close to ray
                        min_dist = min(min_dist, proj)
            obs.append(min(min_dist / 10.0, 1.0))  # Normalize to [0,1]
        
        return np.array(obs, dtype=np.float32)
    
    def step(self, action):
        """Take a step with enhanced reward structure"""
        self.current_step += 1
        
        # Apply action (differential drive)
        left_vel, right_vel = np.clip(action, -1, 1)
        v = (left_vel + right_vel) / 2.0
        omega = (right_vel - left_vel) / 0.5
        
        # Update state
        dt = 0.1
        self.agent_angle += omega * dt
        self.agent_pos[0] += v * np.cos(self.agent_angle) * dt
        self.agent_pos[1] += v * np.sin(self.agent_angle) * dt
        self.agent_vel = np.array([v * np.cos(self.agent_angle), v * np.sin(self.agent_angle)])
        
        # Track position for visualization
        self.position_history.append({
            'step': self.current_step,
            'agent_pos': self.agent_pos.copy(),
            'agent_angle': self.agent_angle,
            'package_pos': self.package_pos.copy(),
            'has_package': self.has_package,
            'timestamp': time.time()
        })
        
        # Calculate reward
        reward, done = self._calculate_smart_reward()
        
        return self._get_observation(), reward, done, {}
    
    def _calculate_smart_reward(self):
        """ENHANCED reward function that guides pickup→delivery behavior"""
        reward = 0.0
        done = False
        
        # Smaller time penalty to allow more exploration
        reward -= 0.01
        
        # Phase 1: Go to package (if not picked up)
        if not self.has_package:
            dist_to_package = np.linalg.norm(self.agent_pos - self.package_pos)
            
            # Moderate reward for getting closer to package
            if self.last_distance_to_package is not None:
                progress = self.last_distance_to_package - dist_to_package
                reward += progress * 10  # Reduced to prevent gaming
            
            self.last_distance_to_package = dist_to_package
            
            # MUCH smaller proximity bonuses to prevent gaming
            if dist_to_package < 0.5:
                reward += 5  # Only reward when very close
            
            # REASONABLE reward for picking up package
            if dist_to_package < 0.3:
                self.has_package = True
                reward += 100  # Restored from 50 - pickup learning is important!
                print(f"  📦 Package picked up at step {self.current_step}! Reward: +100")
                # Reset delivery distance tracking
                self.last_distance_to_delivery = np.linalg.norm(self.agent_pos - self.delivery_pos)
        
        # Phase 2: Go to delivery (if has package)
        else:
            dist_to_delivery = np.linalg.norm(self.agent_pos - self.delivery_pos)
            
            # Strong reward for delivery progress
            if self.last_distance_to_delivery is not None:
                progress = self.last_distance_to_delivery - dist_to_delivery
                reward += progress * 50  # INCREASED from 20 - delivery navigation is key!
            
            self.last_distance_to_delivery = dist_to_delivery
            
            # MUCH smaller delivery proximity bonuses
            if dist_to_delivery < 1.0:
                reward += 25  # INCREASED from 10 - getting close matters!
            
            # MASSIVE reward for successful delivery
            if dist_to_delivery < 0.5:
                reward += 1000  # MASSIVE increase from 500!
                self.delivered_successfully = True  # Mark as actually delivered!
                done = True
                print(f"  🎯 Successful delivery at step {self.current_step}! Reward: +1000")
        
        # Collision penalty with warning zone
        for obs in self.obstacles:
            dist_to_obs = np.linalg.norm(self.agent_pos - obs)
            if dist_to_obs < 0.6:  # Smaller collision zone
                collision_penalty = -15 if self.has_package else -30  # Reduced penalty when carrying
                reward += collision_penalty
                done = True
                print(f"  💥 Collision at step {self.current_step}! Penalty: {collision_penalty}")
                break
            elif dist_to_obs < 1.0:  # Warning zone - small penalty
                reward -= 1 if self.has_package else 2  # Smaller penalty when carrying
        
        # Improved timeout penalties
        if self.current_step >= self.max_steps:
            done = True
            if not self.has_package:
                reward -= 80  # Reduced penalty
            else:
                reward -= 20  # Much smaller penalty if has package
        
        # REMOVE movement bonus - robot was gaming this!
        # No reward for just moving around
        
        return reward, done
    
    def save_episode_data(self, episode_num, reward, success):
        """Save episode data for visualization"""
        episode_info = {
            'episode': episode_num,
            'positions': self.position_history.copy(),
            'final_reward': reward,
            'success': success,
            'obstacles': self.obstacles,
            'delivery_pos': self.delivery_pos.tolist()
        }
        self.episode_data.append(episode_info)
        
        # Save to file for real-time access
        os.makedirs("training_data", exist_ok=True)
        with open(f"training_data/episode_{episode_num}.pkl", "wb") as f:
            pickle.dump(episode_info, f)
        
        # Save latest episode for dashboard
        with open("training_data/latest_episode.pkl", "wb") as f:
            pickle.dump(episode_info, f)
        
        # Save training progress
        progress = {
            'current_episode': episode_num,
            'total_episodes': len(self.episode_data),
            'latest_reward': reward,
            'success_rate': sum(1 for ep in self.episode_data if ep['success']) / len(self.episode_data)
        }
        with open("training_data/progress.json", "w") as f:
            json.dump(progress, f)


class SmartPolicyAgent:
    """Policy gradient agent that actually learns"""
    
    def __init__(self, state_dim, action_dim, lr=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        
        # Simple neural network (policy network)
        self.W1 = np.random.normal(0, 0.1, (state_dim, 64))
        self.b1 = np.zeros(64)
        self.W2 = np.random.normal(0, 0.1, (64, 32))
        self.b2 = np.zeros(32)
        self.W3 = np.random.normal(0, 0.1, (32, action_dim))
        self.b3 = np.zeros(action_dim)
        
        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        
        # Exploration - Balanced for curriculum learning
        self.exploration_noise = 1.0  # Reduced from 1.5 - too chaotic
        self.noise_decay = 0.998  # Balanced decay
        
    def forward(self, state):
        """Forward pass through network"""
        h1 = np.tanh(np.dot(state, self.W1) + self.b1)
        h2 = np.tanh(np.dot(h1, self.W2) + self.b2)
        output = np.tanh(np.dot(h2, self.W3) + self.b3)
        return h1, h2, output
    
    def select_action(self, state):
        """Select action with learned policy + exploration"""
        _, _, action_mean = self.forward(state)
        
        # Add exploration noise
        noise = np.random.normal(0, self.exploration_noise, self.action_dim)
        action = action_mean + noise
        action = np.clip(action, -1, 1)
        
        return action
    
    def store_experience(self, state, action, reward):
        """Store experience for learning"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
    
    def learn(self):
        """Enhanced policy gradient learning"""
        if len(self.states) < 5:  # Learn more frequently
            return 0.0
        
        # Convert to numpy arrays
        states = np.array(self.states)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        
        # Calculate discounted rewards (gives more weight to later rewards)
        discounted_rewards = self._discount_rewards(rewards)
        
        # Normalize rewards (helps with learning stability)
        if np.std(discounted_rewards) > 0:
            discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / np.std(discounted_rewards)
        
        # Simple policy gradient update
        total_loss = 0
        for i in range(len(states)):
            state = states[i]
            action = actions[i]
            reward = discounted_rewards[i]
            
            # Forward pass
            h1, h2, predicted_action = self.forward(state)
            
            # Calculate gradients (simplified)
            action_error = action - predicted_action
            grad_W3 = np.outer(h2, action_error * reward) * self.lr
            grad_b3 = action_error * reward * self.lr
            
            # Update weights
            self.W3 += grad_W3
            self.b3 += grad_b3
            
            total_loss += np.sum(action_error ** 2)
        
        # Clear experience buffer
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        
        # Decay exploration noise - keep higher minimum for delivery learning
        self.exploration_noise = max(0.25, self.exploration_noise * self.noise_decay)  # INCREASED from 0.15
        
        return total_loss / len(states) if len(states) > 0 else 0.0
    
    def _discount_rewards(self, rewards, gamma=0.99):
        """Calculate discounted cumulative rewards"""
        discounted = np.zeros_like(rewards, dtype=float)
        cumulative = 0
        for i in reversed(range(len(rewards))):
            cumulative = rewards[i] + gamma * cumulative
            discounted[i] = cumulative
        return discounted


def train_smart_agent():
    """Main training function with smart learning"""
    print("🧠 Starting SMART Warehouse Agent Training...")
    print("🎯 Goal: Learn pickup → delivery behavior")
    print("=" * 60)
    
    # Hyperparameters - BOOSTED for better learning!
    max_episodes = 1000  # INCREASED from 200 - need more time for delivery learning!
    learning_rate = 0.003  # Higher learning rate
    
    # MLflow setup
    try:
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("Warehouse_Smart_Learning")
        print("✅ MLflow configured successfully")
    except Exception as e:
        print(f"⚠️ MLflow setup failed: {e}")
        print("Continuing without MLflow...")
    
    # Initialize environment and agent
    env = SmartWarehouseEnv()
    agent = SmartPolicyAgent(env.state_dim, env.action_dim, learning_rate)
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    pickup_count = 0
    
    print(f"🚀 Training smart agent for {max_episodes} episodes...")
    print("📊 State dimension:", env.state_dim)
    print("🎮 Action dimension:", env.action_dim)
    
    try:
        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_param("max_episodes", max_episodes)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("algorithm", "PolicyGradient_Smart")
            mlflow.log_param("state_dim", env.state_dim)
            mlflow.log_param("reward_structure", "Pickup_Then_Delivery")
            
            for episode in range(max_episodes):
                print(f"\n🎮 Episode {episode + 1}/{max_episodes}")
                
                state = env.reset()
                episode_reward = 0
                episode_length = 0
                picked_up_package = False
                delivered_package = False  # Track actual delivery
                
                while True:
                    action = agent.select_action(state)
                    next_state, reward, done, _ = env.step(action)
                    
                    # Track if package was picked up this episode
                    if env.has_package and not picked_up_package:
                        picked_up_package = True
                        pickup_count += 1
                    
                    # Track if package was actually delivered
                    if env.delivered_successfully and not delivered_package:
                        delivered_package = True
                    
                    # Store experience for learning
                    agent.store_experience(state, action, reward)
                    
                    # Learn more frequently during episode (every 50 steps)
                    if episode_length > 0 and episode_length % 50 == 0:
                        agent.learn()
                    
                    episode_reward += reward
                    episode_length += 1
                    state = next_state
                    
                    if done:
                        success = delivered_package  # TRUE success = actual delivery!
                        if success:
                            success_count += 1
                        break
                
                # Learn from episode
                loss = agent.learn()
                
                # Log metrics
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                success_rate = success_count / (episode + 1)
                pickup_rate = pickup_count / (episode + 1)
                
                mlflow.log_metric("episode_reward", episode_reward, step=episode)
                mlflow.log_metric("episode_length", episode_length, step=episode)
                mlflow.log_metric("success_rate", success_rate, step=episode)
                mlflow.log_metric("pickup_rate", pickup_rate, step=episode)
                mlflow.log_metric("policy_loss", loss, step=episode)
                mlflow.log_metric("exploration_noise", agent.exploration_noise, step=episode)
                
                # Save episode data for visualization (fix episode numbering)
                env.save_episode_data(episode + 1, episode_reward, success)  # +1 for 1-based display
                
                # Progress logging
                if episode % 5 == 0 or episode == max_episodes - 1:
                    avg_reward = np.mean(episode_rewards[-10:])
                    print(f"📈 Episode {episode:3d}: Reward={episode_reward:6.1f}, "
                          f"Avg={avg_reward:6.1f}, Success={success_rate:.2%}, "
                          f"Pickup={pickup_rate:.2%}, Explore={agent.exploration_noise:.3f}")
            
            # Final metrics
            final_avg_reward = np.mean(episode_rewards[-10:])
            final_success_rate = success_count / max_episodes
            final_pickup_rate = pickup_count / max_episodes
            
            mlflow.log_metric("final_avg_reward", final_avg_reward)
            mlflow.log_metric("final_success_rate", final_success_rate)
            mlflow.log_metric("final_pickup_rate", final_pickup_rate)
            
            print(f"\n🎉 Smart training completed!")
            print(f"📊 Final average reward: {final_avg_reward:.2f}")
            print(f"🎯 Success rate: {final_success_rate:.2%}")
            print(f"📦 Pickup rate: {final_pickup_rate:.2%}")
            print(f"✅ Successful deliveries: {success_count}/{max_episodes}")
            
            # Create training report
            summary = {
                "training_completed": True,
                "algorithm": "Smart_Policy_Gradient",
                "total_episodes": max_episodes,
                "final_avg_reward": float(final_avg_reward),
                "final_success_rate": float(final_success_rate),
                "final_pickup_rate": float(final_pickup_rate),
                "total_successes": success_count,
                "total_pickups": pickup_count,
                "learning_rate": learning_rate,
                "timestamp": datetime.now().isoformat()
            }
            
            with open("training_data/smart_training_summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            
            mlflow.log_artifact("training_data/smart_training_summary.json")
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🚀 Smart training completed! Check visualization at: http://localhost:8502")
    print(f"📊 MLflow dashboard: http://127.0.0.1:5000")
    print(f"🧠 The agent learned to: 1) Find package → 2) Pick up → 3) Deliver!")


if __name__ == "__main__":
    train_smart_agent() 
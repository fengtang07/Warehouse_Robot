import pybullet as p
import pybullet_data
import numpy as np
import time


class WarehouseEnv:
    """
    A class to represent the warehouse environment in PyBullet.
    """

    def __init__(self, connection_mode=p.GUI):
        """
        Initializes the warehouse environment.

        Args:
            connection_mode: The PyBullet connection mode (e.g., p.GUI for graphical interface, p.DIRECT for non-graphical).
        """
        self.client = p.connect(connection_mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # Load environment assets
        self.plane_id = p.loadURDF("plane.urdf")
        self.agent_id = self.load_agent()
        self.load_obstacles()
        self.delivery_bay_id = self.load_delivery_bay()
        self.package_id = self.load_package()

        # Define action and observation space
        self.action_space_dim = 2  # [left_wheel_velocity, right_wheel_velocity]
        self.observation_space_dim = 20  # Lidar + position + velocity

        self.reset()

    def load_agent(self):
        """Loads the AGV agent into the simulation."""
        start_pos = [0, 0, 0.1]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        # A simple cube URDF can be created for the agent
        agent_urdf_path = "assets/agent.urdf"
        with open(agent_urdf_path, "w") as f:
            f.write("""
            <robot name="simple_agent">
              <link name="base_link">
                <visual>
                  <geometry>
                    <box size="0.5 0.5 0.2"/>
                  </geometry>
                  <material name="blue">
                    <color rgba="0 0 0.8 1"/>
                  </material>
                </visual>
                <collision>
                  <geometry>
                    <box size="0.5 0.5 0.2"/>
                  </geometry>
                </collision>
                <inertial>
                  <mass value="1"/>
                  <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
                </inertial>
              </link>
            </robot>
            """)
        return p.loadURDF(agent_urdf_path, start_pos, start_orientation)

    def load_obstacles(self):
        """Loads shelves as obstacles into the simulation."""
        shelf_urdf_path = "assets/shelf.urdf"
        with open(shelf_urdf_path, "w") as f:
            f.write("""
            <robot name="shelf">
              <link name="base_link">
                <visual>
                  <geometry>
                    <box size="2 0.5 1"/>
                  </geometry>
                   <material name="grey">
                    <color rgba="0.5 0.5 0.5 1"/>
                  </material>
                </visual>
                <collision>
                  <geometry>
                    <box size="2 0.5 1"/>
                  </geometry>
                </collision>
              </link>
            </robot>
            """)
        p.loadURDF(shelf_urdf_path, [3, 2, 0.5])
        p.loadURDF(shelf_urdf_path, [-3, -2, 0.5])

    def load_delivery_bay(self):
        """Loads the delivery bay goal area."""
        bay_urdf_path = "assets/bay.urdf"
        with open(bay_urdf_path, "w") as f:
            f.write("""
            <robot name="delivery_bay">
              <link name="base_link">
                <visual>
                  <geometry>
                    <box size="1 1 0.1"/>
                  </geometry>
                  <material name="green">
                    <color rgba="0 0.8 0 1"/>
                  </material>
                </visual>
                <collision>
                  <geometry>
                    <box size="1 1 0.1"/>
                  </geometry>
                </collision>
              </link>
            </robot>
            """)
        return p.loadURDF(bay_urdf_path, [5, 5, 0.05])

    def load_package(self):
        """Loads a package to be picked up."""
        package_urdf_path = "assets/package.urdf"
        with open(package_urdf_path, "w") as f:
            f.write("""
            <robot name="package">
              <link name="base_link">
                <visual>
                  <geometry>
                    <box size="0.2 0.2 0.2"/>
                  </geometry>
                   <material name="red">
                    <color rgba="0.8 0 0 1"/>
                  </material>
                </visual>
                <collision>
                  <geometry>
                    <box size="0.2 0.2 0.2"/>
                  </geometry>
                </collision>
                <inertial>
                  <mass value="0.1"/>
                  <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
                </inertial>
              </link>
            </robot>
            """)
        return p.loadURDF(package_urdf_path, [0, 2, 0.1])

    def get_observation(self):
        """
        Gets the current state of the agent.

        Returns:
            A numpy array representing the agent's state.
        """
        pos, ori = p.getBasePositionAndOrientation(self.agent_id)
        vel, ang_vel = p.getBaseVelocity(self.agent_id)

        # Lidar-like sensor
        lidar_ranges = []
        for i in range(16):
            angle = i * (2 * np.pi / 16)
            ray_from = list(pos)
            ray_to = [pos[0] + 5 * np.cos(angle), pos[1] + 5 * np.sin(angle), pos[2]]
            ray_result = p.rayTest(ray_from, ray_to)
            lidar_ranges.append(ray_result[0][2])

        return np.concatenate([pos, p.getEulerFromQuaternion(ori), vel, ang_vel, lidar_ranges])

    def step(self, action):
        """
        Applies an action to the agent and steps the simulation.

        Args:
            action: A numpy array with the velocities for the left and right wheels.

        Returns:
            A tuple containing the next observation, the reward, whether the episode is done, and additional info.
        """
        # Simplified differential drive
        left_wheel_velocity, right_wheel_velocity = action
        base_pos, base_ori = p.getBasePositionAndOrientation(self.agent_id)
        _, _, yaw = p.getEulerFromQuaternion(base_ori)

        v = (left_wheel_velocity + right_wheel_velocity) / 2
        omega = (right_wheel_velocity - left_wheel_velocity) / 0.5  # Assuming 0.5 is the wheel base

        vx = v * np.cos(yaw)
        vy = v * np.sin(yaw)

        p.resetBaseVelocity(self.agent_id, linearVelocity=[vx, vy, 0], angularVelocity=[0, 0, omega])
        p.stepSimulation()

        next_observation = self.get_observation()
        reward, done = self.calculate_reward()

        return next_observation, reward, done, {}

    def calculate_reward(self):
        """Calculates the reward for the current state."""
        pos, _ = p.getBasePositionAndOrientation(self.agent_id)
        package_pos, _ = p.getBasePositionAndOrientation(self.package_id)
        delivery_pos, _ = p.getBasePositionAndOrientation(self.delivery_bay_id)

        dist_to_package = np.linalg.norm(np.array(pos) - np.array(package_pos))
        dist_to_delivery = np.linalg.norm(np.array(package_pos) - np.array(delivery_pos))

        # Reward logic
        reward = -0.1  # Time penalty
        done = False

        if dist_to_package < 0.3:
            # "Pick up" the package by attaching it to the agent
            p.changeConstraint(
                p.createConstraint(self.agent_id, -1, self.package_id, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
                                   [0, 0, 0.3]))
            reward += 20

        if dist_to_delivery < 0.5:
            reward += 100
            done = True

        # Collision penalty
        if len(p.getContactPoints(self.agent_id,
                                  self.plane_id)) == 0:  # Check for collisions with objects other than the plane
            if len(p.getContactPoints(self.agent_id)) > 0:
                reward -= 10
                done = True

        return reward, done

    def reset(self):
        """Resets the environment to its initial state."""
        p.resetBasePositionAndOrientation(self.agent_id, [0, 0, 0.1], p.getQuaternionFromEuler([0, 0, 0]))
        p.resetBasePositionAndOrientation(self.package_id, [0, 2, 0.1], p.getQuaternionFromEuler([0, 0, 0]))
        p.resetBaseVelocity(self.agent_id, [0, 0, 0], [0, 0, 0])
        return self.get_observation()

    def close(self):
        """Closes the PyBullet connection."""
        p.disconnect(self.client)


if __name__ == '__main__':
    # Example usage
    env = WarehouseEnv()
    for _ in range(1000):
        action = np.random.uniform(-1, 1, size=2)
        obs, reward, done, info = env.step(action)
        time.sleep(1. / 240.)
        if done:
            env.reset()
    env.close()

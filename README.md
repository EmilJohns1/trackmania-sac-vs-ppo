# Progress

Initially, after creating the SAC (Soft Actor-Critic) agent, we noticed it wasn’t learning effectively. We concluded that our model either wasn’t complex enough or didn’t have the computational power to train on the TM20Full environment, which has a high-dimensional observation space consisting of speed, gear, RPM, and four 64 x 64 images. This led us to switch to the TM20Lidar environment, which instead provides 4 x 19 Lidar beams. With this reduced observation space, we could process the inputs using a Multilayer Perceptron (MLP) for the Actor and Critics.

After switching to this environment, we observed an improvement in learning, though the agent still wasn’t performing at the level we expected. We then experimented with various hyperparameters:
- **Alpha**: Reduced from 0.20 to 0.10
- **Target Entropy**: Adjusted from -3.0 to -1.5
- **Gradient Norm Clipping**: Decreased from 1.0 to 0.5
- **Constant Time Penalty**: Reduced from 1.0 to 0.1

Additionally, we revised our reward system. We introduced harsher penalties for colliding with or driving close to walls, as the agent had a tendency to drive alongside the walls. We also implemented a penalty for excessive steering. With these changes, we noticed a significant improvement in the agent’s performance, as shown in the video below:

[![Watch the video](https://img.youtube.com/vi/H-gu15B3E9Y/0.jpg)](https://www.youtube.com/watch?v=H-gu15B3E9Y)

However, over time, the agent’s performance began to degrade. We hypothesized that this was due to the excessive penalty for colliding with walls combined with an insufficient constant time penalty. This caused the agent to drive very slowly to avoid wall collisions, which is illustrated in the video and graph below:

![Performance Graph](readme/graphs/policy_drift.png)
[![Watch the video](https://img.youtube.com/vi/WVIzBIZRctk/0.jpg)](https://www.youtube.com/watch?v=WVIzBIZRctk)

To address this, we adjusted the reward system again. We increased the constant time penalty from 0.1 to 0.2 and decreased the collision penalty from -10 to -8. We also increased the penalty for excessive steering from -0.5 to -1.0 to prevent the agent from steering erratically, as observed in earlier runs.

These adjustments aim to balance speed and safety, enabling the agent to drive more effectively within the constraints of the environment.

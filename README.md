# Hybrid CBS Project with RL and A*

This repository contains an implementation of a hybrid CBS (Conflict-Based Search) planner that integrates reinforcement learning (RL) and A* as low-level planners. The project allows for both single-agent and multi-agent path planning, leveraging the strengths of different methods.

## Files

- **RL_train.py**  
  Train the RL planner (Rainbow DQN) for single-agent scenarios.

- **RL_test.py**  
  Test the performance of the trained RL planner in single-agent planning tasks.

- **RL_CBS.py**  
  Test the performance of the RL-integrated CBS framework in multi-agent scenarios, using RL as the low-level planner.

- **CBS.py**  
  Test CBS with flexible low-level planners (RL and A*), demonstrating the hybrid multi-agent planning capability.

- **best_25.98.pth**  
  Pre-trained model weights for the RL planner.

## How to Run

1. **Train the RL Planner**  
   ```bash
   python RL_train.py

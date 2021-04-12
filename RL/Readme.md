# RL Overview

### What is Reinforcement Learning? 

Reinforcement learning (RL) is an optimization algorithm that interacts with an environment and performs actions in this environment to get the desired result. For example, in a video game, the player moves and performs actions in order to win the game. RL can be applied to such environments (the video game) and can learn to take actions that will result in the player beating the game. To distinguish it from other optimizers such as the genetic algorithm, if the environment demands it, the RL can learn to take a sequence of actions that yield good results. This is often the case in many environments such as videogames and robotics. 

### How is Reinforcement Learning being applied in the Fluid Pinball? 

For our project, we are applying RL in a fluid system and training it to achieve active flow control. Specifically, the Fluid Pinball is the system we wish to control.  The Fluid Pinball has three cylinders that causes fluid instability as fluid flows past the cylinders. However, the three cylinders can rotate (actuate) to suppress these instabilities. The issue is finding a proper actuation control strategy that will reduce the instability and do so without using too much motor power. This is where the RL comes in. The RL is trained in the Fluid Pinball and learns through trial and error an actuation strategy for the three cylinders that meet the desired objectives. 

In RL, we start with an initial random policy. This policy is not expected to produce good results since it randomly assigns an action for a state input (using a neural network). Training is done using episodes to improve the policy such that actions that result in better cumulative rewards are taken. After a set number of episodes, the policy is updated by updating the neural network using stochastic gradient decent. The value assesses whether the action performed better than the current policy, and the actions which produced good results relative to the current policy are made more probable in the future. This improves the overall policy.  At the start, the actions taken are random since the RL is exploring the environment. With learning, the RL starts taking actions with more confidence and with less randomness.

### What RL type is used? 

There are many different types of RL algorithms. The one used here is a policy based RL called PPO. It supports continuous actions and having a large range of state inputs, making it suitable for the Fluid Pinball. 

### Credits

The following GitHub repository served as a basis to develop our code:
https://github.com/mandrakedrink/PPO-pytorch/blob/master/ppo/ppo.py#L30

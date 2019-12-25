[//]: # (Image References)

[image1]: DDPG.png "DDPG training"


# Project 2: Continuous Control

For this project I decided to use a DDPG implementation developed to solve the `Pendulum` OpenAI Gym environment, and then trying to improve it, introducing variations like multiagent training and gradient clipping.

## Learning Algorithm: DDPG

The Deep Deterministic Policy Gradient algorithm is a particular actor-critic method, where the critic is a neural network used to estimate the action value function  Q(s,a) and the actor mu(a) is another neural network that outputs the action maximizing Q. Hence, the training process evolves alternating the following steps:   

- training the Q network, minimizing the temporal difference error, with the actor network parameters fixed.
- training the actor nework mu, maximizing Q(s,mu(s,a)), with the Q network parameter fixed

The high affinity with the DQN algorithm allows to use also the following improvements:

- target networks: in order to avoid instability issues, the expected Q-value at the time step t+1 is calculated using a network which is frozen periodically; the same is used for the critic network.
- experience replay buffer: the learning steps are carried on by mini-batches backpropagations, after sampling randomly from a buffer of memory, in order to avoid that too much correlated transitions drive the process to overfitting; using a multiagent training, the experience of each agent is collected in the same buffer, in order to update the common networks.

The training step has been performed once per timestep, that is, once every 20 experiences, using the target network soft-update (θ_target = τ*θ_local + (1 - τ)*θ_target). This should improve the learning stability. For this purpose, I also used gradient clipping on the Q network gradients.

For exploration, I used the Ornstein-Uhlebeck noise, using a Gaussian sampling.

The actor neural network takes the state as input consists of 3 fully connected layers, with relu activations for the first 2 (hidden) and a tanh activation for the output (which in fact should be in the range (-1,1)).  
The actor neural network consists of 3 fully connected layers, with relu activations for the first 2 (hidden) and a linear activation for the output. The first input, the state, is taken by the fist layer, while the second one, the action, is taken by the second layer together with the ouptut of the first one.

The hyperparameters used are:

- BUFFER_SIZE = int(1e5)  # replay buffer size
- BATCH_SIZE = 128        # minibatch size
- GAMMA = 0.99            # discount factor
- TAU = 1e-3              # for soft update of target parameters
- LR_ACTOR = 1e-4         # learning rate of the actor 
- LR_CRITIC = 1e-3        # learning rate of the critic 

The reward history during training is shown in the following picture:

![DDPG_trained][image1]
# learning rate of the actor 
that is also visible in the `Continuous_Control.ipynb` file together with a verbose logging of average rewards over the last 100 steps: it has taken 289 episodes to solve the problem, that is, in order to get an average reward greater than 30. The related weight files are `checkpoint_actor.pth` and `checkpoint_critic.pth`.


## Possible improvements

The



## Task description

The laboratory activity consist in training an agent to explore an environment modeled as a grid of cells. Each cell has a **level of knowledge** ranging from 0 to 1. The agent has a 
**range of vision**. The knowledge levels of all cells within the vision range will be set to 1, while the remaining ones will be multiplied by a forgetfulness factor.<br>
At each step, the agent can move one cell to the right, left, up or down. The environment is updated accordingly and the sum of the elements of the difference between the new environment 
and the old one will constitute the reward that the agent obtains at that step.<br> 
The state of the system is represented by the **position** (i.e. the row and the column) of the agent in the environment and by the **feature vector** returned by a **Convolutional Neural Network** that takes as input a preprocessed image representing the environment. The state will be the input of a **Deep-Q Network** which will output the Q-values approximations of the four possible actions the agent can take in that state.<br>
During the training, the **ε-greedy policy** will be applied.<br>
The task is modeled as an **episodic task**, therefore each episode has the same finite duration.

## Code structure

Inside the code, there are two different classes:
- the ***DQN*** class defines the structure of the Deep-Q Networks. Specifically, this is a Multi-layer Perceptron with three linear layers. After each layer, except the output one, a
  ReLU function is applied to introduce the non-linearity
- the ***Agent*** class represents the agent. It has several attributes, like the range of vision, the learned and the target networks, the CNN and the deque that memorizes the samples
  for the Deep-Q Networks. There are also several functions:
    - *\_init\_cnn*: initialize the CNN that will be used to get the feature vectors of the environment. That is a pretrained ResNet-18 without the fully connected layer
    - *preprocess\_img*: applies the necessary transformations to the image of the environment in order for it to be processed by the CNN
    - *get\_state*: obtains the state of the system at the current step
    - *choose\_action*: determines the next action that will be performed by the agent
    - *execute\_action*: makes the agent perform the chosen action, or makes it remain still if it would exit the environment
    - *update\_epsilon*: the ε parameter is updated through an exponential decay
    - *remember*: adds to the deque the (state, action, reward, next state) tuple
    - *replay*: performs the training of the learned network by extracting batch size random samples from the deque. The target newtork is used for the targets calculation. The MSE loss
                is applied. Finally, the weights of the target network are updated to those of the learned network if the current instant is a multiple of the Tr attribute

The other functions are:

- *update\_env*: the env is updated based on the agent's new position
- *get\_reward*: calculates the instant reward
- *save\_agent* and *load\_agent*: they are responsible for saving and loading the agent's state respectively so that it doesn't have to be retrained from scratch every time
- *train*: trains the agent for the number of episodes indicated. For each episode, it prints the initial position (which is always random) and the total reward obtained
- *init\_plot* and *update\_animation*: they are used to show an execution of the agent in the environment
- *evaluation*: the agent is tested for a single episode. At each step, the action performed will be always the one that is considered the best. Furthermore, each frame of the execution
                is saved so as to show and save a gif containing the entire execution of the agent

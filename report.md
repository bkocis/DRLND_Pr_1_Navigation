## Project Navigation
Implementation of Deep Q-Learning Algorithms for solving navigation in a small virtual environment and item collection

Content:
1. Approach 
2. Learning algorithms and code implementation 
3. Results of experiments with different settings and hyperparameter tuning 



### 1. Approach  

The project description, installation and requirements are given in the README of the repository.
In order to complete the project, I started from the code provided in the OpenAI Gym lunar landing example. 
I modified the environment related settings on the provided `Navigation.ipynb` notebook. 
I modified the environment relevant sections of the code:

```python
# reset the environment
env_info = env.reset(train_mode=True)[brain_name]  # Udacity env 
#state = env.reset()							   # OpenAI-Gyn env

state = env_info.vector_observations[0]            # get the current state
score = 0 
for t in range(max_t):
	action = agent.act(state, eps)

	# In OpenAI Gym env
	#next_state, reward, done, _ = env.step(action)
	#action = np.random.randint(action_size)        # select an action
	
	# In Unity env
	env_info = env.step(action)[brain_name]        # send the action to the environment
	next_state = env_info.vector_observations[0]   # get the next state
	reward = env_info.rewards[0]                   # get the reward
	done = env_info.local_done[0]                  # see if episode has finished
```
 
### 2. Learning algorithms and code implementation

I reference these sources, as they helped me a lot to get started with the implementation of the algorithms.
- [tommytracey's](https://github.com/tommytracey/DeepRL-P1-Navigation)
- [glebashnik's](https://github.com/glebashnik/udacity-deep-reinforcement-learning-navigation)


#### a). Q-learning

![](https://latex.codecogs.com/svg.latex?Y^{Q}_{t}&space;=&space;R_{t&plus;1}&plus;\gamma&space;maxQ(S_{t&plus;1},&space;a;&space;{\theta}_{t}))

#### b). Deep Q Networks (DQN)

![](https://latex.codecogs.com/svg.latex?Y^{DQN}_{t}&space;=&space;R_{t&plus;1}&plus;\gamma&space;maxQ(S_{t&plus;1},&space;a;&space;{\theta}^{\\-}_{t}))

#### c). Double Q-learning

![](https://latex.codecogs.com/svg.latex?Y^{DoubleQ}_{t}&space;=&space;R_{t&plus;1}&plus;\gamma&space;Q(S_{t&plus;1},&space;argmaxQ(S_{t&plus;1},a;\theta_{t});&space;{\theta}^{'}_{t}))

#### d). Double Deep Q-Network

DDQN differs from Double Q-learning only by the weights of the second network which are replaced with the weights of the target network: 

![](https://latex.codecogs.com/svg.latex?Y^{DoubleDQN}_{t}&space;=&space;R_{t&plus;1}&plus;\gamma&space;Q(S_{t&plus;1},&space;argmaxQ(S_{t&plus;1},a;\theta_{t});&space;{\theta}^{-}_{t}))

The idea of Double Q-learning is to reduce overestimations ... therefore propose to evaluate the greedy policy according to the online network, but using the target network to estimate its value. [Hasselt et al.](https://arxiv.org/pdf/1509.06461.pdf)

#### 4.2 Implementation

Only the learning method of the Agent class is modified. 
Instead defining the next targets of the Q-Network as the maximum value of the targets, the states are gathered and the maximum value assigned to the next target

```python
if self.ddqn:
	indices = torch.argmax(self.qnetwork_local(next_states).detach(),1)
	Q_targets_next = self.qnetwork_target(next_states).detach().gather(1,indices.unsqueeze(1))
else:
	Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

```

#### 5. Deep Learning model 

For the implementation of the DQN, the definition of the Deep learning model is required, and in this examples it is given in the `model.py` file. 
In this case a 3 layer fully connected network was used with ReLU activation functions. 


#### 6. Dueling network

In the rubric a reference to the [Dueling Networks](https://arxiv.org/abs/1511.06581) is provided. 
Implemented in code by modifying the code in `model.py` as follows

```python
# in order to implement dueling the state_value needs to be declared
self.state_value = nn.Linear(fc2_units, 1)
```
and also the forward function in the `QNetwork` was extended so that the final Linear layer in the fully connected network is extended by the state values.

```python 
# use duleling only when required
if self.dueling:
	return self.fc3(x) + self.state_value(x)
else:
	return self.fc3(x)
```



#### 7. Epsilon greedy action selection 

The selection of the next action after a step can be made as a random choice. This would correspond to something like a free exploration. The idea of epsilon greedy algorithm is to contain the selection of the next action based on the epsion value, similar to a threshold value for the random choice of actions. When overexposed, the agent chooses same actions over and over again, which would lead to poor solutions of the environment. 
The values of the epsion can be implemented as changing values with the learning progression. By defining an decay and end value, the epsion value will be updated for each learning step during the agent training. 

The code implementation was present in the example code used from the Lunar landing example and was not modified (in the [`dqn_agent.py`](https://github.com/bkocis/DRLND_Pr_1_Navigation/blob/master/dqn_agent.py#L82#L86).


#### 8. Agent training algorithm 

The 


### 3. Results of experimenta of different settings and hyperparameter tuning 

In the cource of the DQN code implementation I ran several experiments, and only the final ones are included in the `Navigation.ipynb` notebook. 
In order to have some ground for comparison of the experiments, all of them have to be executed in the same session of environment load. Restarting the notebook between experiments would make the comparison of the hyperparameter tuning results false. 

The greatest impact on the speed (number of episodes till desired score of 13 was reached) was due to the epsilon greedy algorithm. The `eps_decay` parameter, that sets basically the speed of the eps value decrease, was modified in a few experiments. The least episodes needed to reach target score were obtained when the eps_decay value was set to 0.98. Lower values did not produce much better results, as the agent becomes quite unstable (deduced empirically by observing the score plot and the large difference of score values varying around the average). Value of eps_decay equal to 1 in this code implementation would mean no change of the epsilon values, means a total random choice of actions, which can be seen on the score plot as well. 

Implementation of dueling and double DQN resulted only in a slight improvement in solving the environment, when both were used together. 

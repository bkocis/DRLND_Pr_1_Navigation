# Project Navigation
## Implementation of Deep Q-Learning Algorithms for solving navigation and item collection in small virtual environment

Content:
1. Approach 
2. Learning algorithms and code implementation 
3. Results of experimenta of different settings and hyperparameter tuning 



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


#### 1. Q-learning

#### 2. Deep Q Networks

#### 3. Double Q-learning

#### 4. Double Deep Q-Network

#### 4.1 Theory

The idea of Double Q-learning is to reduce overestimations ... therefore propose to evaluate the greedy policy according to the online network, but using the target network to estimate its value. [Hasselt et al.](https://arxiv.org/pdf/1509.06461.pdf)
Double DQN-s update is the same as for DQN, but replacing the target `Y^{DQN}_{t}` with `Y^{DoubleDQN}_{t} = R_{t}+1+γQ(St+1, argmaxQ(St+1, a; θt), θ^{−}_{t})`.
Double DQN-s update is the same as for DQN, but replacing the target 

```math 
Y^{DQN}_{t} with Y^{DoubleDQN}_{t} = R_{t}+1+γQ(St+1, argmaxQ(St+1, a; θt), θ^{−}_{t})
```


`^2 + b^2 = c^2$$`

`$$\begin{vmatrix}a & b\\
c & d
\end{vmatrix}=ad-bc$$`


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


#### 5. Dueling network

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


### 3. Results of experimenta of different settings and hyperparameter tuning 

In the cource of the DQN code implementation I ran several experiments, and only the final ones are included in the `Navigation.ipynb` notebook. 
In order to have some ground for comparison of the experiments, all of them have to be executed in the same session of environment load. Restarting the notebook between experiments would make the comparison of the hyperparameter tuning results false. 




The project description, installation and requirements are given in the README of the repository.

In order to complete the project, I started from the code provided in the OpenAI gun lunar landing example. 

1. First I modified the environment related settings on the provided `Navigation.ipynb` notebook. 
2. I on github previous solutions of this project. 
I reference:
- [tommytracey's](https://github.com/tommytracey/DeepRL-P1-Navigation)
- [glebashnik's](https://github.com/glebashnik/udacity-deep-reinforcement-learning-navigation)
 
solution, as it helped me a lot to get started with the implementation of the algorithms.


I modified the environment relevant sections of the code:

```python
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment

        #state = env.reset()
        #state = env_info.vector_observations[0]
        #score = 0
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


# Learning 

1. Dueling network
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

2. Double Deep-Q Network

Only the learning method of the Agent class is modified. 
Instead defining the next targets of the Q-Network as the maximum value of the targets, the states are gathered and the maximum value assigned to the next target

```python
        if self.ddqn:
            indices = torch.argmax(self.qnetwork_local(next_states).detach(),1)
            Q_targets_next = self.qnetwork_target(next_states).detach().gather(1,indices.unsqueeze(1))
        else:
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        
```


started from the code provided in the OpenAI gun lunar landing example. 


I modified the environemnt relevant sections of the code:

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


1. implement dueling network by modifying the code in model.py 

```python
        self.dueling = dueling
        # in order to implement dueling the state_value needs to be declared
        self.state_value = nn.Linear(fc2_units, 1)
```
and

```python 
        # use duleling only when required
        if self.dueling:
            # advantage values + state value
            return self.fc3(x) + self.state_value(x)
        else:
            return self.fc3(x)
```

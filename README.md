# Reinforcement Learning Tools

For this section of the assignment I designed a set of methods which could apply to both Question 3 and Question 4. In this section of the report I will describe the general process for designing these tools and what they do.

## rlTools.simulate()

```python
def simulate(
    env:object, # RL Environment
    alpha:float, # SARSA Training Parameter
    gamma:float, # SARSA Training Parameter
    num_episodes:int, # Number of episodes to run
    num_actions:int, # Number of available actions
    epsilon:float, # Greedy Epsilon Function Parameter
    policy:np.ndarray = np.zeros(1), # Optional Input RL Policy
    cutoff:int = 20000, # Validation Cutoff
    **states:int # Number of states as separate named variables
    ) -> tuple[np.ndarray, list, list]:
```

This method is the overarching tool which utilises most of the other functions I developed. This method can be used to either train or test a reinfocement learning agent for minirace and has been extended to work with minipong.

The main magic behind the tool is the ability to dynamically generate the template reinforcement policy table based on state information provided. The function takes in the number of actions and the expected number of states for each layer of the system as a keyword argument (e.g. 15 for layer 1 and 15 for layer 2). To generate the policy table a state mapping function is used to appropriately index the table.

The function can also take in a already generated policy to either continue training or test an agent with the policy.

The method implements SARSA and greedy epsilon for training and a validation step. If all parameters alpha, gamma and epsilon are set to 0, The method utilises only the validation step to test the policy with no learning.

## rlTools.runEpisode()

```python
def runEpisode(
    env:object,
    policy:np.ndarray,
    training:bool = True,
    record:bool = False,
    epsilon:float = 0,
    alpha:float = None, 
    gamma:float = None,
    cutoff:int = 20000,
    **states:int,
    ) -> (tuple[np.ndarray, int, list, list] or tuple[np.ndarray, int]):
```

This method simulates a single episode of the reinforcement agent environment with the supplied environment and policy.

The method can be run in training mode or testing mode by setting the training flag to true or false respectively.

If no values are provided for alpha or gamma the method defaults to testing mode. If no epsilon is provided the method will only warn the user as it isn't necessary to training.

```python
mode = training
if not alpha and not gamma and mode:
    print("Warning!: No Sarsa learning values provided for training mode.\nDefaulting to testing mode.")
    mode = False

if not epsilon and mode:
    print("Warning!: No Epsilon value provided for training mode.\ngreedy epsilon operator won't randomly choose actions which will affect reinforcement learning.")

if alpha is not None and gamma is not None and not mode:
    print("Warning!: Function has been set to testing mode but Gamma and alpha values have been provided.\nValues will not be used in this simulation.")
```

Episode training runs under this general structure when you exclude all of the conditional statements. The steps of reinforcement learning are outlined in the comments titled with step in this code.

```python

# STEP 1: Initialise the Episode
# Initialise variables
done = False
trunc = False
rewardSum = 0

# Reset environment and get initial state
state = env.reset()

# Map state using state mapping function
state = mapState(state, states)

# Choose Initial Action
action = choose(policy, state, epsilon)


# Start training Episode Run. Stop when done or trunc occurs
while not done and not trunc:

    # STEP 2: Get new state and reward from environment step
    # Step Reinforcement Learning Actor and get state information
    newState, reward, done = env.step(action)

    # STEP 3: Map state information to policy table index
    # Map State using state mapping function
    newState = mapState(newState, states)

    # STEP 4: Choose a new action using greedy epsilon operator
    # Get new action using greedy epsilon choose function
    newAction = choose(policy, newState, epsilon)

    # STEP 5: Update total reward and check for truncation
    # Update run time and check if the system has truncated
    rewardSum += reward
    if rewardSum >= cutoff:
        trunc = True
    reward = float(reward)

    # STEP 6: Use SARSA to update Reinforcement Learning Policy table
    # SARSA state action table update
    policy[state, action] = policy[state, action] + alpha*(reward + gamma*policy[newState, newAction] - policy[state, action])
        
    # STEP 7: set current state and action to the new state and action
    # Update state and action
    state = newState
    action = newAction
    
```

Pixel data from the episode can also be extracted by setting the record flag true. Episode completion will return two extra values, A list of numpy arrays which represent the environment frame and a list of reward data for each frame of the simulation.

## rlTools.mapState()

```python
def mapState(state, states):
```

The mapState method is probably the most crucial function for tools to function. The method is responsible for converting the range of input states to suitable indexes for the policy table this function gets more important as more layers are added to the state output. As the number of input layers increase the number of possible states is equal to the product of all the possible states per layer. These values need to be mapped from a higher dimension index to a 1 dimension table index. The function is set up in a way to take any number of state inputs and correctly index a unique row in the policy table.

```python
# Get Total number of states possible
num_states = np.prod(list(states.values()))

# Get middle value of all state layers to offset input state values
offsets = np.divide(np.subtract(tuple(states.values()),1),2)
state = state+offsets

# Make an array with all integers for the number of possible states then reshape the array to match the dimensions and size of the state layers
stateMap = np.arange(num_states).reshape(tuple(states.values()))

# Get Table state row index based on state provided
mappedState = tuple(state.astype(int))
return stateMap[mappedState]
```

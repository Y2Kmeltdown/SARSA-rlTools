import os
from minirace import Minirace
import numpy as np
import pickle
from datetime import datetime
import cv2 as cv

def choose(q, state, epsilon):
    
    """
    epsilon-greedy action selection
    Parameters
    ----------
    q : np.array (states x actions)
    A q table for each action, state combination
    state : int (0..143)
    The current state of the agent.
    epsilon : float (between 0 and 1)
    Probability of taking a random action.
    Returns
    -------
    int
    The chosen action, representing one of
    env.actions.forward, env.actions.left, env.actions.right
    """
    delta = np.random.rand(1)
    if delta < epsilon:
        return np.random.randint(np.shape(q[state,:])[0])
    else:
        return np.argmax(q[state,:])

def mapState(state, states):
    num_states = np.prod(list(states.values()))
    offsets = np.divide(np.subtract(tuple(states.values()),1),2)
    state = state+offsets
    stateMap = np.arange(num_states).reshape(tuple(states.values()))
    mappedState = tuple(state.astype(int))
    return stateMap[mappedState]

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
    mode = training
    if not alpha and not gamma and mode:
        print("Warning!: No Sarsa learning values provided for training mode.\nDefaulting to testing mode.")
        mode = False

    if not epsilon and mode:
        print("Warning!: No Epsilon value provided for training mode.\ngreedy epsilon operator won't randomly choose actions which will affect reinforcement learning.")

    if alpha is not None and gamma is not None and not mode:
        print("Warning!: Function has been set to testing mode but Gamma and alpha values have been provided.\nValues will not be used in this simulation.")

    # Reset environment and get state
    state = env.reset()
    #Map state using state mapping function
    state = mapState(state, states)
    # Choose Action
    action = choose(policy, state, epsilon)

    done = False
    trunc = False

    rewardSum = 0
    
    videoData = []
    rewardData = []

    # Start training Episode Run. Stop when done or trunc occurs
    while not done and not trunc:

        if record:
            if type(env) == Minirace:
                x, z, _ = env.s1
                pix = env.to_pix(x, z, False)
            else:
                pix = env.to_pix(env.s1)
            videoData.append(pix)
            rewardData.append(rewardSum)
            
        # Step Reinforcement Learning Actor and get state information
        newState, reward, done = env.step(action)
        # Map State using state mapping function
        newState = mapState(newState, states)
        
        # Get Reward information
        rewardSum += reward
        if rewardSum >= cutoff:
            trunc = True
        reward = float(reward)

        if not trunc:
            if mode:
                # Get new action using greedy epsilon choose function
                newAction = choose(policy, newState, epsilon)
                # SARSA state action table update
                policy[state, action] = policy[state, action] + alpha*(reward + gamma*policy[newState, newAction] - policy[state, action])
            else:
                # Get new action using greedy epsilon choose function
                newAction = choose(policy, newState, 0)
            #Update state and action
            state = newState
            action = newAction
    
    if record:
        return policy, rewardSum, videoData, rewardData
    else:
        return policy , rewardSum

def simulate(
    env:object,
    alpha:float, 
    gamma:float,
    num_episodes:int,
    num_actions:int,
    epsilon:float, 
    policy:np.ndarray = np.zeros(1),
    cutoff:int = 20000,
    **states:int
    ) -> tuple[np.ndarray, list]:
    """
    SARSA for our minigrid environment
    Parameters
    ----------
    env : environment
    a gym environment
    alpha : float
    a learning rate.
    epsilon : float
    a probability for choosing a random action (0 <= epsilon <= 1).
    gamma : float
    a discount value (0 < gamma < 1).
    num_episodes : int, optional
    Number of episodes to train. The default is 50.
    num_actions : int, optional
    Number of actions available. The default is 3.
    num_states : int, optional
    Number of states in the environment can be provided as multiple values to define state layers.
    Returns
    -------
    q : array (num_states x num_actions)
    state-action value table.
    """
    #Get Total number of states from system layers
    num_states = np.prod(list(states.values()))
    
    # Check if a policy is provided. If no policy exists initialise an empty policy
    if not policy.any():
        q = 0.333 * np.ones((num_states, num_actions))
    else:
        q = policy

    # Check that provided or generated policy matches state and action constraints
    if np.shape(q) != (num_states, num_actions):
        raise ValueError("Provided Policy Does not match environment constraints")

    trainingReward = []
    testReward = []
    # loop through episodes stated in the function input
    while num_episodes > 0:
        # Print out information every 10 episodes
        if num_episodes % 10 == 0:
            if testReward:
                averageReward = sum(testReward[-11:-1])/10
            else:
                averageReward = 0
            print(f'####Episodes left: {num_episodes} | Average Reward: {averageReward}####',end="\r")
        
        # Training loop
        if epsilon != 0 and gamma != 0 and alpha != 0:
            q, rewardSum = runEpisode(
                env=env,
                policy=q,
                epsilon=epsilon,
                alpha=alpha,
                gamma = gamma,
                cutoff=cutoff,
                **states
            )
            trainingReward.append(rewardSum)

        # Testing Loop
        q, testRewardSum = runEpisode(
            env=env,
            policy=q,
            training=False,
            cutoff=cutoff,
            **states
        )
        testReward.append(testRewardSum)
        
        # If a desired performance is reached stop training
        if epsilon != 0 and gamma != 0 and alpha != 0:
            if testRewardSum >= cutoff:
                print("\n") 
                return q, trainingReward, testReward
        
        # Decrement episodes
        num_episodes -= 1

        
        
        
    print("\n")    
    return q, trainingReward, testReward

def savePolicy(policy:np.ndarray, location:str=None):
    fileName = "rlPolicy_%s.pickle"%datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    if not location:
        fileDir = "trainedPolicies\\"
    else:
        fileDir = location
    if os.path.isdir(fileDir):
        filePath = fileDir + fileName
        with open(filePath, 'wb') as file:
            pickle.dump(policy,file)
    else:
        raise ValueError("Save Failed. No such location exists")

def loadPolicy(fileName:str):
    if os.path.isfile(fileName):
        with open(fileName, "rb") as file:
            policy = pickle.load(file)
        return policy
    else:
        raise ValueError("Load Failed. No such file exists")

def recordVideo(videoData:list, rewardData:list, scale:int = 1):
    print("Frame Data Retrieved")
    step = 0
    height = int(np.shape(videoData[0])[0]*scale)
    width = int(np.shape(videoData[0])[1]*scale)
    frameSize = (height, width)

    videoFile = "videos\\reinforcement_%s.avi"%datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    out = cv.VideoWriter(videoFile,cv.VideoWriter_fourcc(*'MJPG'), 15, frameSize, False)
    print("Writing Video")
    for frame in videoData:
        frame = np.repeat(frame, scale, axis=0)
        frame = np.repeat(frame, scale, axis=1)
        frame = np.asarray(np.multiply(frame,255/2), dtype=np.uint8)
        cv.putText(frame, "Step %i"%(rewardData[step]), (10,height-10), cv.FONT_HERSHEY_SIMPLEX, .5, 255)
        out.write(frame)
        step += 1
    out.release()
    print("Video Complete")

if __name__ == "__main__":
    #q = loadPolicy("trainedPolicies\\WorkingPolicyLevel2.pickle")
    race = Minirace(level=2)

    q, rewardData, testRewardData = simulate(
        env = race,
        alpha = 0.2,
        gamma = 0.90,
        epsilon = 0.01,
        num_episodes = 20000,
        num_actions = 3,
        layer1 = 15,
        layer2 = 15
        )
    

    q, rewardSum, videoData, rewardData = runEpisode(
        env = race,
        policy = q,
        training = False,
        record = True,
        layer1 = 15,
        layer2 = 15
    )

    if rewardData[-1] >= 15000:
        savePolicy(q)
        recordVideo(videoData,rewardData,scale = 16)


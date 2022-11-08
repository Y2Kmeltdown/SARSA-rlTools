#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 14:29:55 2022

@author: oliver
"""

import numpy as np
import random
from matplotlib import pyplot as plt
from itertools import count
import time

class Minirace:
    
    def __init__(self, level = 1, size = 6, normalise = False):
        # level is the dimensionality of state vector (1 or 2)
        self.level = min(2, max(1, level))
        # size is the number of track positions (every 2 pixels, from 1..n-1). 
        # this means there are 2 more car positions (0 and n)
        self.size = max(3, size)
        self.xymax = 2*(self.size + 2)
        # whether to normalise the state representation
        self.scale = 2 if normalise and level > 1 else 1.0        
        
        # the internal state is s1 = (x, z, d). Previous internal state in s0. 
        # x: x-coordinate of the car
        # z[]: x-coordinate of the track, one for each y-coordinate
        # d[]: dx for the track, for each y-coordinate        
        self.reset()
        
        
    def observationspace(self):
        """
        Dimensionality of the observation space

        Returns
        -------
        int
            Number of values return as an observation.

        """
        return self.level
    

    def nexttrack(self, z, d = 2):
        """
        Move the next piece of track based on coordinate z, and curvature

        Parameters
        ----------
        z : int
            x-coordinate for the previous track segment.
        d : int, optional
            previous "curvature" (change of coordinate). The default is 2.
            This is to prevent too strong curvature (car can only move 1 step)

        Returns
        -------
        znext : int
            The x-coordinate for the next piece of track (middle of the track).
        dz : int
            The change compared to the previous coordinate (-2..2).

        """
        trackd = random.randint(-2, 2)        
        if self.level == 1 or abs(d) > 1:
            trackd = min(1, max(-1, trackd))
            
        znext = max(1, min(self.size, z + trackd))
        dz = znext - z
        return znext, dz
    
    
    def state(self):
        """
        Returns the (observed) state of the system.
        
        Depending on level, the observed state is an 
        array of 1 to 5 values, or a pixel representation (level 0).

        Returns
        -------
        np.array
            level 1: [dx] 
                 dx: relative distance in x-coordinate between car and next 
                     piece of track (the one in front of the car).
                     May be normalised to values between -1 and 1, 
                     depending on initialisation.
        """        
        x, z, d = self.s1
        ## level 1:
        # return the difference between car x and the next piece of track
        if self.level == 1:
            return np.array([(z[2] - x) / self.scale])
        if self.level == 2:
            return np.array([(z[2] - x) / self.scale, (z[3] - z[2]) / self.scale])
            
        raise ValueError("level not implemented")
        
                
    def transition(self, action = 0):
        """
        Apply an action and update the environment.
        0: do nothing
        1: move left
        2: move right

        Parameters
        ----------
        action : int, optional
            The action applied before the update. 
            The default is 0 (representing no action).

        Returns
        -------
        np.array
            The new observed state of the environment.

        """
        self.s0 = self.s1

        if self.terminal():
            return self.state()

        x0, z0, d0 = self.s0
        z1 = np.roll(z0, -1)
        d1 = np.roll(d0, -1)

        x1 = x0
        if action == 1:
            x1 = max(0, x0 - 1)
        elif action == 2:
            x1 = min(self.size-1, x0 + 1)
            
        z1[-1], d1[-1] = self.nexttrack(z0[-1], d0[-1]) 
        self.s1 = (x1, z1, d1)
        return self.state()


    def terminal(self):
        """
        Check if episode is finished.

        Returns
        -------
        bool
            True if episode is finished.

        """
        x, z, _ = self.s1
        
        return abs(z[1] - x) > 1.0

    
    def reward(self, action):    
        """
        Calculate immediate reward.
        Positive reward for staying on track.

        Parameters
        ----------
        action : int
            0-2, for the 3 possible actions.

        Returns
        -------
        r : float
            immediate reward.

        """
        r = 1.0 if not self.terminal() else 0.0
                        
        return r
    

    def step(self, action):
        # return tuple (state, reward, done)
        state = self.transition(action)
        r = self.reward(action)
        done = self.terminal()
        return (state, r, done)    
    
    
    def reset(self):
        # the internal state is 
        # x: x-coordinate of the car
        # z[]: x-coordinate of the track, one for each y-coordinate
        # d[]: dx for the track, for each y-coordinate
        x = random.randint(0, self.size-1)
        z = np.zeros(self.xymax)
        d = np.zeros(self.xymax)
        z[0] = max(1, min(self.size, x))
        z[1] = z[0]
        d[1] = 2
        for i in range(2, self.xymax):
            z[i], d[i] = self.nexttrack(z[i-1], d[i-1])
                
        self.s0 = (x, z, d)
        self.s1 = (x, z, d)
        
        return self.state()


    def sampleaction(self):
        # return a random action [0,2]
        action = random.randint(0, 2)
        return action
      
    
    def render(self, text = True, reward = None, cm = plt.cm.bone_r, f = None):
        x, z, _ = self.s1
        pix = self.to_pix(x, z, text)
        if text:
            if reward is not None:
                print('{:.3f}'.format(reward))            
            print(''.join(np.flip(pix, axis=0).ravel())) 
        else:
            fig, ax = plt.subplots()
            if reward is not None:
                plt.title(f'Reward: {reward}', loc='right')
            ax.axis("off")
            plt.imshow(pix, origin='lower' , cmap=cm)
            if f is not None:
                plt.savefig(f, dpi=300)
            plt.show()                    

        
    def to_pix(self, x, z, text = False):
        """
        Generate a picture from an internal state representation 

        Parameters
        ----------
        x : int
            car x-coordinate
        z : np.array
            array with track coordinates            
        text : bool, optional
            flag if generate text represenation
            
        Raises
        ------
        ValueError
            If x,y,z are outside their range.

        Returns
        -------
        image : np.array
            a square image with pixel values 0, 0.5, and 1.

        """
        if x < 0 or x > self.size+1:
            raise ValueError('car coordinate value error')
        if np.min(z) < 1 or np.max(z) > self.size:
            raise ValueError('track coordinate value error')

        car = '#' if text else 2

        if text:
            image = np.array(list(':'*(self.xymax+1))*(self.xymax)).reshape(self.xymax,-1)
            image[:, -1] = '\n'
        else:
            image = np.ones((self.xymax, self.xymax), dtype = int)

        for i, j in enumerate(z):
            j = int(j * 2)
            image[i, j-2:j+4] = ' ' if text else 0

        image[0:2, 2*x:(2*x+2)] = car
        
        return image
      
    
def mypolicy(state):
    #action = therace.sampleaction()
    if state < 0:
        action = 1
    elif state > 0:
        action = 2
    else:
        action = 0
    return action

if __name__ == "__main__":
    seed = 1
    # torch.manual_seed(seed)

    gamma = 0.99
    render = True
    finalrender = True
    log_interval = 100
    render_interval = 1000
    running_reward = 0

    therace = Minirace(level=1, size=6)

    starttime = time.time()

    for i_episode in count(1):
        state, ep_reward, done = therace.reset(), 0, False
        rendernow = i_episode % render_interval == 0
    
        for t in range(1, 10000):  # Don't infinite loop while learning

            # select action (randomly)
            action = mypolicy(state)

            # take the action
            state, reward, done = therace.step(action)
            reward = float(reward)     # strange things happen if reward is an int

            if render and rendernow:
                therace.render(reward = ep_reward)

            ep_reward += reward
            if done:
                break

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # log results
        if i_episode % log_interval == 0:
            print('Episode {}\t Last reward: {:.2f}\t Average reward: {:.2f}'.format(
                i_episode, ep_reward, running_reward))

        # check if we have solved minirace
        if running_reward > 50:
            secs = time.time() - starttime
            mins = int(secs/60)
            secs = round(secs - mins * 60.0, 1)
            print("Solved in {}min {}s!".format(mins, secs))
            
            print("Running reward is now {:.2f} and the last episode "
                  "runs to {} time steps!".format(running_reward, t))

            if finalrender:
                state, ep_reward, done = therace.reset(), 0, False
                for t in range(1, 500):
                    action = mypolicy(state)
                    state, reward, done = therace.step(action)
                    ep_reward += reward
                    therace.render(text = False, reward = ep_reward)
                    if done:
                        break
        
            break
        
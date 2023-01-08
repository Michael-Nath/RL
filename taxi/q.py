'''
file: q.py
Description:
This program contains an implementation of the quintessential Q-learning algorithm.
We will be applying this algorithm to the OpenAI Taxi gym environment, which is 
noted to be very simple, and posess discrete states. Thus this allows us to work with 
tabular representations of our values (action values in the case of unknown MDP).
'''

import gymnasium as gym
import numpy as np

class TaxiQLearningAgent:
    def __init__(self, EPS_COUNT = 1000) -> None:
        # init Taxi environment
        self.env = gym.make("Taxi-v3")
        self.n_actions = self.env.action_space.n
        self.n_states = self.env.observation_space.n
        self.EPS_COUNT = EPS_COUNT
        # training Hyperparameters
        self.EPSILON = 0.2
        self.GAMMA = 0.3
        self.ALPHA = 0.99
        # optimistic initialization of q values
        self.q_pi = np.ones((self.n_states, self.n_actions))
    
    def choose_action_epsilon_greedy(self, state: int):
        '''
        Epsilon-soft policy for choosing an action.
        Return: 
        action: an action for the car (down, up, right, left, pick-up, drop-off)
        '''
        if np.random.random() < self.EPSILON:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.q_pi[state])
        return action
    
    def run(self):
        '''
        We will be using naive Q-Learning to update the Q-values. 
        '''
        for _ in range(self.EPS_COUNT):
            # first we reset the environment
            state, _ = self.env.reset()
            # we loop until termination (or more likely, truncation)
            while True: # guaranteed to terminate because truncation occurs at 200th timestep
                # we follow the action told by our policy (stochastic at first -> deterministic once optimal)
                action = self.choose_action_epsilon_greedy(state)
                # now we take this action
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                # temporal-difference learning
                self.q_pi[state, action] = (1 - self.ALPHA) * self.q_pi[state, action] + self.ALPHA * (reward + self.GAMMA * np.max(self.q_pi[next_state]))
                state = next_state
                if terminated or truncated:
                    break
        return self.q_pi
    
if __name__ == "__main__":
    agent = TaxiQLearningAgent()
    q_pi = agent.run()
    np.savetxt("q-values-q.txt", q_pi)
    agent.env.close()
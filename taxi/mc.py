'''
file: mc.py
Description:
This program contains an implementation of on-policy monte carlo control.
We will be applying this algorithm to the OpenAI Taxi gym environment, which is 
noted to be very simple, and posess discrete states. Thus this allows us to work with 
tabular representations of our values (action values in the case of unknown MDP).
'''

from collections import defaultdict
from typing import Tuple, List
import gymnasium as gym
import numpy as np

class TaxiMonteCarloAgent:

    def __init__(self, EPS_COUNT = 500_000) -> None:
        # init Taxi environment
        self.env = gym.make("Taxi-v3")
        self.n_actions = self.env.action_space.n
        self.n_states = self.env.observation_space.n
        self.EPS_COUNT = EPS_COUNT
        # training hyperparameters
        self.EPSILON = 0.2
        self.GAMMA = 0.3
        # optimistic Initialization of Q-Values
        self.q_pi = np.ones((self.n_states, self.n_actions))
        # keep track of first-visits to each state-action pair
        self.N : defaultdict[Tuple[int, int], int] = defaultdict(lambda: 0.0)

    def choose_action_epsilon_greedy(self, state: int) -> int:
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

    def generate_episode(self) -> List[Tuple[int, int, int]]:
        '''
        Upon invocation, `generate_episode` will roll-out one episode — the policy evaluation step.
        S_0, A_0, R_1, ..., S_{T-1}, A_{T_1}, R_T
        
        Return:
        history: a collection of entries of the form (s_t, a_,t, r_{t+1}). This is used during the policy improvement step.
        '''
        state, _ = self.env.reset()
        history = []
        while True: # guaranteed to terminate because truncation occurs at 200th timestep
            # we follow the action told by our policy (stochastic at first -> deterministic once optimal)
            action = self.choose_action_epsilon_greedy(state)
            # now we take this action
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            history.append((state, action, reward))
            state = next_state
            if terminated or truncated:
                break
        return history
    def update(self, history):
        '''
        Once an episode is generated, `update` is called to update the state-action values based on the most recent trajectory.
        This is effectively the policy improvement step.
        '''
        rewards = np.array([t[-1] for t in history])
        for t in range(len(history)):
            state, action, _ = history[t]
            if (state, action) in history[:t]:
                continue
            G = rewards[t:].sum()
            self.N[(state, action)] += 1
            self.q_pi[state, action] += (G - self.q_pi[state, action]) / self.N[(state, action)]

    def run(self):
        '''
        `run` performs policy iteration (policy evaluation + policy improvement).
        '''
        for ep in range(self.EPS_COUNT):
            history = self.generate_episode() # policy evaluation
            self.update(history) # policy improvement
        return self.q_pi


if __name__ == "__main__":
    agent = TaxiMonteCarloAgent()
    q_pi = agent.run()
    np.savetxt("q-values.txt", q_pi)
    agent.env.close()
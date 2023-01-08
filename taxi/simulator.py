import gymnasium as gym
import numpy as np
from typing import Union
GAME = "Taxi-v3" # currently algorithms are only tested for Taxi game

class GymSimulator:
    def __init__(self, visualize=True, q_values_fname: Union[str, None] = None):
        self.env = gym.make(GAME, render_mode="human") if visualize else gym.make(GAME)
        self.q_values_fname = q_values_fname
        
    def simulate(self, supplied_q_values: Union[np.ndarray, None], n_times = 10):
        assert (supplied_q_values is not None or self.q_values_fname is not None)
        if supplied_q_values is not None:
            assert supplied_q_values.shape == (self.env.observation_space.n, self.env.action_space.n)
        q = supplied_q_values if self.q_values_fname is None else np.loadtxt(self.q_values_fname)
        successes = 0.0
        for _ in range(n_times):
            state, _ = self.env.reset()
            while True:
                action = np.argmax(q[state])
                next_state, _, terminated, truncated, _ = self.env.step(action)
                state = next_state
                if terminated or truncated:
                    if terminated:
                        successes += 1
                    break
        print(f"Sucesss Rate (Successes / Trials) = {successes} / {n_times} = {successes * 100 / n_times}%")
        
    # TODO: generate some useful statistics (cumulative regret, cumulative reward)
    def gen_statistics(self):
        pass
    
    # TODO: generate policy plots
    def gen_plots(self):
        pass
    
if __name__ == "__main__":
    print("Determining Monte Carlo Agent Performance...")
    mc_simulator = GymSimulator(visualize=False, q_values_fname="q-values.txt")
    mc_simulator.simulate(None, n_times=10000)
    mc_simulator.env.close()
    print("Determining Q-Learning Agent Performance...")
    q_simulator = GymSimulator(visualize=False, q_values_fname="q-values-q.txt")
    q_simulator.simulate(None, n_times=10000)
    q_simulator.env.close()
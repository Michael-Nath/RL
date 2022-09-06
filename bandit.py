import numpy as np


class KArmedBandit:
    def __init__(self, k:int, EPS_PROBABILITY=0.1, TIME_STEPS=10000, perturb=False):
        self.k = k
        self.EPS_PROBABILITY = EPS_PROBABILITY
        self.TIME_STEPS = TIME_STEPS
        self.perturb = perturb
    
    def step_size_fn(self, a: int) -> float:

        return 1 / self.n[a]
    
    def perform_one_run(self, equal_values=False):
        # Init run data so we can plot learning progression
        self.run_data = np.array([])
        # Init accuracy vector so we can plot percent optimal action taken
        self.accuracy = np.array([], dtype=float)
        # Init Q(a) for all a
        self.q = np.full(self.k, 0, dtype=float)
        # Init N(a) for all a
        self.n = np.zeros(self.k)
        # Init q*(a) for all a
        if equal_values:
            self.q_star = np.full(self.k, np.random.normal(0, 1))
        else:
            self.q_star = np.random.randn(self.k)    
        for t in range(1, self.TIME_STEPS + 1):
            a = self.choose_action(t)
            if (a >= np.max(self.q_star)):
                self.accuracy = np.append(self.accuracy, 1)
            else:
                self.accuracy = np.append(self.accuracy, 0)
            self.perform_one_time_step(a)
    def choose_action(self, _) -> int:
        # Epsilon greedy policy, pick randomly with probability epsilon, otherwise pick greedily
        epsilon = np.random.random()
        if epsilon < self.EPS_PROBABILITY:
            a = np.random.randint(0, self.k)
        else:
            a = np.argmax(self.q)
        return a
    def perform_one_time_step(self, a: int):
        reward = self.bandit(a)
        # Incrementally Computed Value Method
        self.q[a] = self.q[a] + self.step_size_fn(a) * (reward - self.q[a])
        self.run_data = np.append(self.run_data, self.q[a])
    def bandit(self, a):     
        '''
        bandit(action)
        Given an action performed at timestep i, this function returns a reward r_i taken out of a probability distribution with q*(a) as mean and variance 1.
        However at each time step, all q*(a) get perturbed by a value picked out of a normaml with mean 0 and stdev 0.01
        '''   
        # Pulling this 'lever' adds 1 to the total times pulled for this action
        self.n[a] += 1
        # Perturb q*(a) if enabled
        perturbation = 0
        if self.perturb:
            perturbation = np.random.normal(0, 0.01)
        self.q_star += perturbation
        # return a reward picked out of this normal distribution
        return np.random.normal(self.q_star[a], 1)

class ExpRecencyBandit(KArmedBandit):
    ALPHA = 0.1
    def step_size_fn(self, _) -> float:
        """ Exponential Recency step size function puts greater weight on recent rewards.

        Args:
            _ (int): Blank argument since step size function operates irrespective of the action taken

        Returns:
            float: The step size
        """
        return self.ALPHA
class UCBBandit(KArmedBandit):
    C = 2
    def choose_action(self, t: int) -> int:
        epsilon = np.random.random()
        if epsilon < self.EPS_PROBABILITY:
            a = np.random.randint(0, self.k)
        else:
            a = np.argmax(self.q + self.C * np.sqrt(np.log(t) / self.n))
        return a
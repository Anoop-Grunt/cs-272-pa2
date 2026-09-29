import numpy as np
import gymnasium as gym
from gymnasium import spaces

from myagent import SarsaLambdaAgent


class OneStepEnv(gym.Env):
    observation_space = spaces.Discrete(2)
    action_space = spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return 0, {}

    def step(self, action):
        return 1, 1.0, True, False, {}


env = OneStepEnv()

agent = SarsaLambdaAgent(
    env,
    alpha=0.5,
    gamma=0.99,
    eps=0.0,
    lam=0.0,
    total_epi=1,
    init_val=0.0,
    seed=0,
)

before = agent.q.copy()
agent.learn()
after = agent.q.copy()

changed_entries = np.argwhere(before != after)

print("Changed entries:", changed_entries)
print("Q-table before:\n", before)
print("Q-table after:\n", after)

assert len(changed_entries) == 1
print("λ=0 check passed")

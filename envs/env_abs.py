import gym
from gym.utils import seeding
from abc import ABC, abstractmethod


class AbsEnv(gym.Env, ABC):

    @abstractmethod
    def __init__(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @property
    @abstractmethod
    def action_space(self):
        pass

    @property
    @abstractmethod
    def observation_space(self):
        pass


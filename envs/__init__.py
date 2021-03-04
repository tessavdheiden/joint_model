import gym
gym.logger.set_level(40)

from envs.env_ball_box import BallBoxEnv
from envs.env_arm import ArmEnv
from envs.env_reacher import ReacherEnv
from envs.env_controlled_reacher import ReacherControlledEnv
from envs.env_arm_follow_rectangle import ArmFollowRectangleEnv
from envs.env_tanh2d import Tanh2DEnv
from envs.env_pendulum import PendulumEnv
from envs.env_sigmoid import SigmoidEnv
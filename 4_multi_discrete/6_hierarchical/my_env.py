import os
import itertools

import numpy as np

import gym
from gym import error, spaces
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel


class Action:
    def __init__(self, action):
        class Continuous:
            def __init__(self):
                self.shape = (1, 0)

            def __getitem__(self, i):
                return 0
            
        self.continuous = Continuous()
        self.discrete = action.reshape([1, -1])


class MyEnv(gym.Env):
    def __init__(self, worker_id, realtime_mode=False):
        self.reset_parameters = EnvironmentParametersChannel()
        self.engine_config = EngineConfigurationChannel()

        env_path = "C:/myDesktop/source/gridworld_imitation/food_collector_4"

        self._env = UnityEnvironment(env_path, worker_id, side_channels=[self.reset_parameters, self.engine_config])
        self._env.reset()

        self.behavior_name = list(self._env.behavior_specs)[0]
        behavior_spec = self._env.behavior_specs[self.behavior_name]
        print(behavior_spec)

        if realtime_mode:
            self.engine_config.set_configuration_parameters(time_scale=1.0)
            self.reset_parameters.set_float_parameter("train-mode", 0.0)
        else:
            self.engine_config.set_configuration_parameters(time_scale=20.0)
            self.reset_parameters.set_float_parameter("train-mode", 1.0)

        self._flattener = ActionFlattener(behavior_spec.action_spec.discrete_branches)

    def reset(self):
        # for key, value in reset_params.items():
        #     self.reset_parameters.set_float_parameter(key, value)
        self._env.reset()
        info, terminal_info = self._env.get_steps(self.behavior_name)
        self.game_over = False

        obs, reward, done, info = self._single_step(info, terminal_info)
        return obs

    def step(self, action):
        # Use random actions for all other agents in environment.
        if self._flattener is not None and type(action) == int:
            # Translate action into list
            action = np.array(self._flattener.lookup_action(action))
        
        c_action = Action(action)

        self._env.set_actions(self.behavior_name, c_action)
        self._env.step()
        running_info, terminal_info = self._env.get_steps(self.behavior_name)
        obs, reward, done, info = self._single_step(running_info, terminal_info)
        self.game_over = done

        return obs, reward, done, info

    def _single_step(self, info, terminal_info):
        if len(terminal_info) == 0:
            done = False
            use_info = info
        else:
            done = True
            use_info = terminal_info
            
        # 카메라, 센서 순으로 나옴
        output_info = {}
        output_info["visual_obs"] = use_info.obs[0][0]

        #obs = np.concatenate([use_info.obs[1][0], use_info.obs[2][0]])     
        return use_info.obs[1][0], use_info.reward[0], done, output_info

    def close(self):
        self._env.close()

    def render(self):
        pass


class ActionFlattener:
    """
    Flattens branched discrete action spaces into single-branch discrete action spaces.
    """

    def __init__(self, branched_action_space):
        """
        Initialize the flattener.
        :param branched_action_space: A List containing the sizes of each branch of the action
        space, e.g. [2,3,3] for three branches with size 2, 3, and 3 respectively.
        """
        self._action_shape = branched_action_space
        self.action_lookup = self._create_lookup(self._action_shape)
        self.action_space = spaces.Discrete(len(self.action_lookup))
        print(self.action_lookup)

    @classmethod
    def _create_lookup(self, branched_action_space):
        """
        Creates a Dict that maps discrete actions (scalars) to branched actions (lists).
        Each key in the Dict maps to one unique set of branched actions, and each value
        contains the List of branched actions.
        """
        possible_vals = [range(_num) for _num in branched_action_space]
        all_actions = [list(_action) for _action in itertools.product(*possible_vals)]
        
        # Dict should be faster than List for large action spaces
        action_lookup = {
            _scalar: _action for (_scalar, _action) in enumerate(all_actions)
        }
        return action_lookup

    def lookup_action(self, action):
        """
        Convert a scalar discrete action into a unique set of branched actions.
        :param: action: A scalar value representing one of the discrete actions.
        :return: The List containing the branched actions.
        """
        return self.action_lookup[action]

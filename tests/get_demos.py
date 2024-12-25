from __future__ import annotations

import gymnasium as gym
import argparse

import pytest
print(gym.__file__)
from minigrid.utils.baby_ai_bot import BabyAIBot
import logging
logger = logging.getLogger(__name__)
import sys
import subprocess
import os
import time
import numpy as np
import blosc
import torch
import pdb
import random
import copy
import matplotlib.pyplot as plt
from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper, SymbolicObsWrapper

from gymnasium.wrappers import RecordVideo
import pandas as pd
df_positions = pd.DataFrame(columns=['a'])
import warnings

# Suppress all user warnings
warnings.filterwarnings("ignore", category=UserWarning)


import babyai.utils as utils
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--env",
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", default='BOT',
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--demos", default=None,
                    help="path to save demonstrations (based on --model and --origin by default)")
parser.add_argument("--episodes", type=int, default=1000000,
                    help="number of episodes to generate demonstrations for")
parser.add_argument("--valid-episodes", type=int, default=512,
                    help="number of validation episodes to generate demonstrations for")
parser.add_argument("--seed", type=int, default=0,
                    help="start random seed")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected")
parser.add_argument("--log-interval", type=int, default=100,
                    help="interval between progress reports")
parser.add_argument("--save-interval", type=int, default=2500,
                    help="interval between demonstrations saving")
parser.add_argument("--filter-steps", type=int, default=0,
                    help="filter out demos with number of steps more than filter-steps")
parser.add_argument("--on-exception", type=str, default='warn', choices=('warn', 'crash'),
                    help="How to handle exceptions during demo generation")

parser.add_argument("--job-script", type=str, default=None,
                    help="The script that launches make_agent_demos.py at a cluster.")
parser.add_argument("--jobs", type=int, default=0,
                    help="Split generation in that many jobs")

args = parser.parse_args()

# see discussion starting here: https://github.com/Farama-Foundation/Minigrid/pull/381#issuecomment-1646800992
broken_bonus_envs = {
    "BabyAI-PutNextS5N2Carrying-v0",
    "BabyAI-PutNextS6N3Carrying-v0",
    "BabyAI-PutNextS7N4Carrying-v0",
    "BabyAI-KeyInBox-v0",
}

# get all babyai envs (except the broken ones)
babyai_envs = []
for k_i in gym.envs.registry.keys():
    if k_i.split("-")[0] == "BabyAI":
        if k_i not in broken_bonus_envs:
            babyai_envs.append(k_i)
babyai_envs = ['BabyAI-KeyCorridorS6R3-v0']
# breakpoint()
def print_demo_lengths(demos):
    num_frames_per_episode = [len(demo[2]) for demo in demos]
    logger.info('Demo length: {:.3f}+-{:.3f}'.format(
        np.mean(num_frames_per_episode), np.std(num_frames_per_episode)))

def update_positions_df(a):
    global df_positions
    # Create a new DataFrame for the current row to append
    new_row = pd.DataFrame({'a':[a]})
    # Use pandas.concat to append the new row
    df_positions = pd.concat([df_positions, new_row], ignore_index=True, axis=0)
@pytest.mark.parametrize("env_id", babyai_envs)
def test_bot(env_id):
    """
    The BabyAI Bot should be able to solve all BabyAI environments,
    allowing us therefore to generate demonstrations.
    """
    # Use the parameter env_id to make the environment
    print(env_id)
    env = gym.make(env_id,) # for visual debugging
    env = gym.make(env_id, render_mode="rgb_array", highlight = False) # for visual debugging
    env = FullyObsWrapper(env)
    env = SymbolicObsWrapper(env)
    # reset env
    curr_seed = 1

    num_steps = 500
    terminated = False
    n_episodes = 1 #2000
    count = 0
    just_crashed = False
    demos_path = utils.get_demos_path(args.demos, env_id, 'agent', False)
    demos = []
    while True:
        if count == n_episodes:
            break
        print(f"Episode {count}")
        
        terminated = False
        if just_crashed:
            logger.info("reset the environment to find a mission that the bot can solve")
            env.reset(seed=curr_seed + count)
        for pos in range(1):
            terminated = False
            obs, _ = env.reset(seed=curr_seed + count)
            env = RecordVideo(env, f'./video_goto_mini/video_goto_pos_{pos}')
            while not terminated:
                # create expert bot
                
                expert = BabyAIBot(env)

                last_action = None
                all_goals = expert.get_all_goal_state()
                all_goals = all_goals[:2] + all_goals[-2:]
                # breakpoint()    
                goal = all_goals
                actions = []
                # breakpoint()
                mission = obs["mission"]
                print(mission)
                full_obs = env.gen_full_obs()
                rgb_image = obs['image']
                goal_state = None
                cur_states = []
                next_states = []
                images = []
                rgb_images = []
                action_temp = [None]*6
                for _step in range(num_steps):
                    action = expert.replan(last_action)
                    cur_state = list(env.agent_pos) + [obs['direction']] + goal
                    cur_states.append(cur_state)
                    obs, reward, terminated, truncated, info = env.step(action)
                    update_positions_df(action)
                    last_action = action
                    images.append(full_obs)
                    next_state = list(env.agent_pos) + [obs['direction']] + goal
                    next_states.append(next_state)
                    actions.append(action)
                    full_obs = env.gen_full_obs()
                    obs = obs

                    if reward > 0:
                        demos.append((mission, blosc.pack_array(np.array(images)), cur_states, actions, next_states))
                        just_crashed = False
                    if terminated:
                        break

                # try again with a different seed
                # curr_seed += 1
            env.close()
        count += 1
        
    logger.info("Saving demos...")   
    # breakpoint()
    utils.save_demos(demos, demos_path)
    logger.info("{} demos saved".format(count))
    print_demo_lengths(demos[-100:])
    df_positions.to_csv('action_s.csv')
        
if __name__ == "__main__":
    test_bot('BabyAI-OpenDoorsOrderN4-v0')
    # test_bot('BabyAI-GoToLocalS10N1-v0')

    # test_bot('MiniGrid-PutNear-6x6-N2-v0')
    
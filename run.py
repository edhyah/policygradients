#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Scripts that initiate training.
Not intended for production use.
"""

import argparse

import gym
import matplotlib.pyplot as plt
import numpy as np

import models
import ppo
import vpg
import utils

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', choices=['vpg', 'ppo'], default='ppo',
            help='Algorithm to train CartPole policy.')
    parser.add_argument('--seed', type=int, default=0,
            help='Seed for random number generator.')
    parser.add_argument('--max-episodes', type=int, default=2000,
            help='Maximum number of episodes to train on.')
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()

    np.random.seed(args.seed)
    env = gym.make('CartPole-v0')
    env.seed(args.seed)

    theta = np.random.rand(*env.observation_space.shape)
    policy = models.LogisticPolicy(theta)

    if args.algo == 'vpg':
        algo = vpg.VPG(env=env, policy=policy, max_episodes=args.max_episodes)
        episode_returns = algo.train()
    elif args.algo == 'ppo':
        algo = ppo.PPO(env=env, policy=policy, max_episodes=args.max_episodes)
        episode_returns = algo.train()
    else:
        raise ValueError('Algorithm called %s does not exist' % args.algo)

    # Test trained policy
    utils.run_episode(env, policy, render=True)
    plt.figure()
    plt.plot(episode_returns)
    plt.title('Episode returns over training')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.show()

if __name__ == '__main__':
    main()


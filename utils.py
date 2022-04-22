#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility functions.
Not intended for production use.
"""

import numpy as np

def discount_rewards(rewards, gamma):
    """Return list of discounted returns.

    Given a list of rewards [r_0, r_1, ..., r_{T-2}, r_{T-1}], return a list
    containing the discounted return at each time step. Formally, this means we
    return a list with each element G_t equal to

        G_t = Σ_{t'=t}^{T-1} gamma^{t'-t} * r_t'

    where gamma is the discount factor.
    """
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for i in reversed(range(0, len(rewards))):
        cumulative_rewards = cumulative_rewards * gamma + rewards[i]
        discounted_rewards[i] = cumulative_rewards
    return discounted_rewards

def run_episode(env, policy, render=False):
    """Obtain a trajectory from the current policy."""
    observations = []
    actions = []
    rewards = []
    probs = []
    total_reward = 0
    done = False

    observation = env.reset()
    while not done:
        if render:
            env.render()

        observations.append(observation)
        action, prob = policy.act(observation)
        observation, reward, done, info = env.step(action)
        total_reward += reward

        rewards.append(reward)
        actions.append(action)
        probs.append(prob)

    rewards = np.array(rewards)
    observations = np.array(observations)
    actions = np.array(actions)
    probs = np.array(probs)

    return total_reward, rewards, observations, actions, probs


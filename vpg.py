#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Vanilla policy gradient implementation.
Not intended for production use.
"""

import numpy as np

from models import MultilayerPerceptron
from utils import discount_rewards, run_episode

class VPG():
    """Vanilla policy gradient.

    Our objective function that we want to maximize is E[R(tau)], or the
    expected value of the return of trajectory tau with respect to the policy.
    Using the Policy Gradient Theorem, it can be mathematically shown that the
    gradient of this function takes the form

        ∇ E[R(tau)] = E[ Σ_{t=0}^{T-1} ∇ log pi(a_t|s_t) R_t ]

    where R is the total discounted reward following action a_t, or

        R_t = Σ_{t'=t}^{T-1} gamma^(t'-t) * R(s_t', a_t')

    We estimate the expectation in the policy gradient via Monte Carlo
    (ie. we take N rollouts and take the average gradient of the N rollouts).
    Then, we perform gradient ascent.

    Attributes:
        env: Gym environment.
        policy: Policy network (ie. actor network).
        lr: Learning rate.
        gamma: Discount factor.
        use_baseline: Use baseline for variance reduction.
        max_episodes: Maximum number of episodes to train.
        value: Value network (ie. critic network).
    """

    def __init__(
            self,
            env,
            policy,
            lr=0.002,
            gamma=0.99,
            use_baseline=True,
            max_episodes=2000):
        self.env = env
        self.policy = policy
        self.lr = lr
        self.gamma = gamma
        self.use_baseline = use_baseline
        self.max_episodes = max_episodes
        self.value = MultilayerPerceptron(
                input_units=env.observation_space.shape[0], output_units=1,
                hidden_units=4, num_hidden=1, lr=lr)

    def train(self):
        episode_returns = []
        for i in range(self.max_episodes):
            total_reward, rewards, observations, actions, probs = run_episode(
                self.env, self.policy)
            episode_returns.append(total_reward)
            self.update(rewards, observations, actions)
            if i % 100 == 0:
                print('Episode #' + str(i) + '\tScore: ' + str(total_reward))
        return episode_returns

    def update(self, rewards, obs, actions):
        grad_log_p = np.array([self.policy.grad_log_p(ob)[action]
            for ob, action in zip(obs, actions)])
        discounted_returns = discount_rewards(rewards, self.gamma)
        values = self.value.predict(obs)

        # If using baseline for variance reduction, subtract value estimates
        # from empirical return. Else just use empirical return.
        if self.use_baseline:
            gradient = grad_log_p.T @ (discounted_returns - values.T).ravel()
        else:
            gradient = grad_log_p.T @ discounted_returns

        # Actor training step
        self.policy.theta += self.lr * gradient

        # Critic training step
        value_loss = self.value.loss(values, discounted_returns[:, None])
        self.value.update(values, discounted_returns[:, None])


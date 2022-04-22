#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Proximal policy optimization implementation.
Not intended for production use.
"""

import numpy as np

from models import MultilayerPerceptron
from utils import discount_rewards, run_episode

class PPO():
    """Proximal policy optimization.

    PPO is an improvement to VPG, in that the core algorithm is still VPG, but
    policy updates are clipped to prevent destructively large policy updates.
    In PPO, we optimize the objective function

        L(theta) = E[ min(r_t(theta)A_t, clip(r_t(theta), 1-eps, 1+eps)A_t) ]

    where r_t is the likelihood ratio from importance sampling, or

        r_t = __pi_new(a_t|s_t)__
                pi_old(a_t|s_t)

    and epsilon is the clipping ratio (a hyperparameter). The use of importance
    sampling allows us to train more efficiently by taking multiple gradient
    steps for each trajectory (as opposed to VPG, where each trajectory only
    allows one gradient step).

    The gradient of r_t(theta)A_t can be derived like so:

        ∇ r_t(theta)A_t = ∇ __pi_new(a_t|s_t)__ . A_t
                              pi_old(a_t|s_t)
                        = __pi_new(a_t|s_t)__ . __∇ pi_new(a_t|s_t)__ . A_t
                            pi_old(a_t|s_t)         pi_new(a_t|s_t)
                        = __pi_new(a_t|s_t)__ ∇ log pi_new(a_t|s_t) A_t
                            pi_old(a_t|s_t)

    Note that because of the min and clip functions in the loss function, we
    the gradient is non-zero only under the following conditions:

        - A_t > 0, r_t < 1 + eps
        - A_t < 0, r_t > 1 - eps

    Otherwise, the gradient of the loss function is zero, and is ignored. Once
    we compute the gradients, we perform gradient ascent to maximize our
    objective function.

    Attributes:
        env: Gym environment.
        policy: Policy network (ie. actor network).
        lr: Learning rate.
        gamma: Discount factor.
        eps: PPO clipping factor.
        max_episodes: Maximum number of episodes to train.
        num_epochs: Number of epochs to train each PPO iteration.
        mini_batch_size: Mini batch size for each gradient update.
        use_gae: Use GAE for advantage estimation.
        lam: Lambda parameter for GAE-lambda.
        value: Value network (ie. critic network).
    """

    def __init__(
            self,
            env,
            policy,
            lr=0.0005,
            gamma=0.99,
            eps=0.2,
            max_episodes=2000,
            num_epochs=4,
            mini_batch_size=5,
            use_gae=False,
            lam=0.95):
        self.env = env
        self.policy = policy
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.max_episodes = max_episodes
        self.num_epochs = num_epochs
        self.mini_batch_size = mini_batch_size
        self.use_gae = use_gae
        self.lam = lam
        self.value = MultilayerPerceptron(
                input_units=env.observation_space.shape[0], output_units=1,
                hidden_units=4, num_hidden=1, lr=lr)

    def train(self):
        episode_returns = []
        for i in range(self.max_episodes):
            total_reward, rewards, observations, actions, probs = run_episode(
                self.env, self.policy)
            discounted_returns = discount_rewards(rewards, self.gamma)
            values = self.value.predict(observations).ravel()

            # Compute advantage using GAE
            if self.use_gae:
                finalstate = self.env.step(actions[-1])[0]
                finalvalue = self.value.predict(finalstate)
                advantages = self.compute_adv(rewards, values, finalvalue)
            else:
                advantages = discounted_returns - values

            episode_returns.append(total_reward)
            self.update(discounted_returns, observations, actions, probs,
                    advantages)
            if i % 100 == 0:
                print('Episode #' + str(i) + '\tScore: ' + str(total_reward))
        return episode_returns

    def update(self, discounted_returns, obs, actions, probs, advantages):
        batch_size = obs.shape[0]
        for _ in range(self.num_epochs):
            for _ in range(batch_size // self.mini_batch_size):
                rand_ids = np.random.randint(0, batch_size, self.mini_batch_size)
                o = obs[rand_ids, :]
                a = actions[rand_ids]
                advs = advantages[rand_ids]
                d_r = discounted_returns[rand_ids]
                p_old = probs[rand_ids]
                p_new = np.choose(a, self.policy.probs(o))
                ratio = p_new / p_old

                # Actor training step
                gradient_ids = self.take_gradient(ratio, advs)
                grad_log_p = np.array([self.policy.grad_log_p(ob)[action]
                    for ob, action in zip(o, a)])
                gradient = (gradient_ids * ratio * grad_log_p.T) @ advs
                self.policy.theta += self.lr * gradient

                # Critic training step
                v = self.value.predict(o)
                value_loss = self.value.loss(v, d_r[:, None])
                self.value.update(v, d_r[:, None])

    def compute_adv(self, rewards, values, finalvalue):
        """Compute advantages using Generalized Advantage Estimation (GAE).

        Args:
            rewards: Array of rewards from trajectory.
            values: Array of value estimates from trajectory.
            finalvalue: Value estimate at time T in a trajectory from 0 to T-1.
        """
        gae = 0
        adv = []
        values = np.append(values, finalvalue)

        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i+1] - values[i]
            gae = delta + self.gamma * self.lam * gae
            adv.append(gae)

        adv.reverse()
        return np.array(adv)

    def take_gradient(self, ratio, advantage):
        """Return a binary vector denoting which indices have non-zero
        gradients.

        For example, a return value of [False, False, True] means the gradient
        at indices 0 and 1 are zero, and the gradient at index 2 is non-zero.
        With this in mind, only the gradient at index 2 must be computed.

        For information on how we can determine if a gradient will be non-zero,
        see the class header comment.
        """
        case1_ids = np.logical_and(advantage > 0, ratio < 1 + self.eps)
        case2_ids = np.logical_and(advantage < 0, ratio > 1 - self.eps)
        gradient_ids = np.logical_or(case1_ids, case2_ids)
        return gradient_ids


# Policy gradients tutorial

A set of example implementations of vanilla policy gradient (VPG) and proximal
policy optimization (PPO). The CartPole environment from OpenAI Gym is used here
to test these algorithm implementations. All model parameterizations are derived
here (no autograd use from deep learning libraries).

## Why?

I understood the theory of RL, understood how to use RL with autograd, but never
truly understood what it means to take a policy gradient. I'm sure others are in
a similar boat, so I've created this tutorial to derive vanilla policy gradient
and PPO algorithms from scratch with my own (small) neural network
implementation.

## Preliminaries

We assume a finite-horizon discounted Markov decision process (MDP) defined by
the tuple (S, A, P, R, rho, gamma), where S is a finite set of states, A is a
finite set of actions, P: S x A x S -> |R is the transition probability
distribution, R: S -> |R is the reward function, rho: S -> |R is the initial
state distribution of the initial state s_0 and gamma in (0, 1) is the discount
factor. Note that |R is the set of real numbers.

Let pi denote a stochastic policy pi: S x A -> [0, 1] that we want to optimize.
Let A_t be the advantage, or Q(s_t, a_t) - V(s_t) where Q and V are the
action-value function and the value function, respectively.

## Installation

Code was only tested on Python 3.8. For other versions, you may have to play
around with dependencies a bit.

First, create a virtual environment:
```
cd /path/to/policygradients
python3 -m venv venv
source venv/bin/activate
```

Then, install dependencies.
```
pip install -r requirements.txt
```

If dependency installation runs into errors, try upgrading pip.
```
pip install --upgrade pip
```

If the pyglet module is causing an error, remove line 9 (`pyglet==1.5.11`) from
requirements.txt and run the following:
```
pip install -r requirements.txt
pip install pyglet==1.5.11
```
Ignore the error about dependency clashes.

That should be it!

## Usage

For usage help, use the following command:
```
python run.py -h
```

Make sure your virtual environment is sourced.

## Bugs

- PPO training with GAE doesn't seem to work yet.

## Additional Resources

- [Original policy gradent paper](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)
- [PPO paper](https://arxiv.org/pdf/1707.06347.pdf)

## Found this useful?

Follow me on [Twitter](https://twitter.com/edwardahn9)!


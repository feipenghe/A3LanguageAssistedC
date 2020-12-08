import argparse
import sys, os, time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch import distributions
from torch.distributions import Categorical
from itertools import islice

import gym
from rlhw_util import * # <-- look whats inside here - it could save you a lot of work!


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        #         self.fc1 = nn.Sequential( nn.Linear(state_dim, state_dim), nn.ReLU())
        self.fc1 = nn.Sequential(nn.Linear(state_dim, state_dim * 8), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(state_dim * 8, state_dim * 4), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(state_dim * 4, state_dim), nn.ReLU())
        self.fc4 = nn.Sequential(nn.Linear(state_dim, action_dim), nn.Softmax())
        # TODO: Fill in the code to define you policy

    def forward(self, state):
        """
        Takes state information and output action distribution.
        """

        # TODO: Fill in the code to run a forward pass of your policy to get a distribution over actions (HINT: probabilities sum to 1)
        x = self.fc1(state)
        x = self.fc2(x)

        x = self.fc3(x)
        x = self.fc4(x)
        return x

    def get_policy(self, state):
        """
        Get a action distribution given the state input
        """
        return Categorical(self(state))

    def get_action(self, state, greedy=None):
        """
        Get rollouts.
        """
        if greedy is None:
            greedy = not self.training

        policy = self.get_policy(state)
        return MLE(policy) if greedy else policy.sample()


class Critic(nn.Module):
    """
    Evaluation the action the actor takes.
    Q learning agent
    """

    def __init__(self, state_dim):
        super(Critic, self).__init__()

        # TODO: define your value function network
        self.fc1 = nn.Sequential(nn.Linear(state_dim, state_dim*2), nn.ReLU())
        self.fc2 = nn.Sequential( nn.Linear(state_dim*2, state_dim), nn.ReLU())
        self.fc3 = nn.Linear(state_dim, 1)

    def forward(self, state):
        # TODO: apply your value function network to get a value given this batch of states
        x = self.fc1(state)
        #         x = self.fc2(x)
        x = self.fc3(x)
        return x


class A3C(nn.Module):

    def __init__(self, state_dim, action_dim, discount=0.97, lr=1e-3, weight_decay=1e-4):
        super(A3C, self).__init__()
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)

        # TODO: create an optimizer for the parameters of your actor (HINT: use the passed in lr and weight_decay args)
        # (HINT: the actor and critic have different objectives, so how many optimizers do you need?)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, weight_decay=weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr, weight_decay=weight_decay)
        self.discount = discount

    def forward(self, state):
        return self.actor.get_action(state)

    def learn(self, states, actions, rewards):
        print("learn1")
        returns = [compute_returns(rs, discount=self.discount) for rs in rewards]
        print("learn2")
        states, actions, returns = torch.cat(states), torch.cat(actions), torch.cat(returns)

        # TODO: implement A3C (HINT: algorithm details found in A3C paper supplement)
        # (HINT2: the algorithm is actually very similar to REINFORCE, the only difference is now we have a critic, what might that do?)

        # update critic
        self.critic_optimizer.zero_grad()
        print("learn3")
        error = F.mse_loss(self.critic(states).squeeze(), returns)

        error.backward()  # update critic

        self.critic_optimizer.step()

        # update actor
        advantage = returns - self.critic(states).squeeze()
        loss = 0
        for i in range(len(states)):
            m = self.actor.get_policy(states[i])
            loss -= m.log_prob(actions[i]) * advantage[i]
        loss /= len(states)
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return error.item()


#         solve(states, returns, out=self.critic)  # solve linear system and update weights?

#         raise NotImplementedError


def run_iteration(mode, N, agent, gen, horizon=None, render=True):
    train = mode == 'train'
    agent = agent.cuda()
    if train:
        agent.train()
    else:
        agent.eval()

    # run N times and generate
    states, actions, rewards = zip(*[gen(horizon=horizon, render=render) for _ in range(N)])

#     states, actions, rewards = states.cuda(), actions.cuda(), rewards.cuda()
    loss = None
    if train: # don't compute loss for evaluation mode
        loss = agent.learn(states, actions, rewards)

    # averaged reward
    reward = sum([r.sum() for r in rewards]) / N
    # r is the sequence [1,1,1] representing it persists 3 seconds
    #

    return reward, loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="MontezumaRevenge-ram-v0", type = str)
    parser.add_argument("--test", default=False, type = bool)
    args = parser.parse_args()

    env_name = args.env
    # env_name = 'LunarLander-v2'
    e = Pytorch_Gym_Env(env_name)
    state_dim = e.observation_space.shape[0]
    action_dim = e.action_space.n

    if args.test:
        pass # TODO: implement testing
    else:
        # Optimization hyperparameters
        lr = 1e-3
        weight_decay = 1e-4

        # env_name = 'CartPole-v1'
        # env_name = 'LunarLander-v2'
        env_name = "MontezumaRevenge-ram-v0"
        e = Pytorch_Gym_Env(env_name)
        state_dim = e.observation_space.shape[0]
        action_dim = e.action_space.n

        # Choose what agent to use
        a3c_agent = A3C(state_dim, action_dim, lr=lr, weight_decay=weight_decay)
        # rl_ll_agent = A3C(state_dim, action_dim, lr=lr, weight_decay=weight_decay)

        total_episodes = 0
        print(a3c_agent)  # Let's take a look at what we're working with...

        # Create a
        gen = Generator(e, a3c_agent)
        rl_ll_xs = []
        rl_ll_ys = []
        num_iter = 50
        num_train = 10
        num_eval = 10  # dont change this

        for itr in range(num_iter):
            # agent.model.epsilon = epsilon * epsilon_decay ** (total_episodes / epsilon_decay_episodes)
            # print('** Iteration {}/{} **'.format(itr+1, num_iter))

            train_reward, train_loss = run_iteration('train', num_train, a3c_agent, gen)
            eval_reward, _ = run_iteration('eval', num_eval, a3c_agent, gen)
            rl_ll_xs.append(total_episodes)
            rl_ll_ys.append(train_reward)
            total_episodes += num_train

            print(total_episodes, train_reward, train_loss, eval_reward)
            print('Ep:{}: reward={:.3f}, loss={:.3f}, eval={:.3f}'.format(total_episodes, train_reward, train_loss,
                                                                          eval_reward))

            if eval_reward > 499 and env_name == 'CartPole-v1':  # dont change this
                print('Success!!! You have solved cartpole task! Time for a bigger challenge!')
                break

            # save model
            torch.save(a3c_agent.state_dict(), "./model")
        print('Done')
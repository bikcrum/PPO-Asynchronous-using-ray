import logging
from datetime import datetime

import gym
from torch.distributions import Categorical

import wandb

from network import MyNetwork
from ppo_discrete import PPO_Discrete
import torch


def train():
    env = gym.make('CartPole-v1')
    ppo = PPO_Discrete(policy_class=MyNetwork,
                       env=env,
                       device_infer=device_infer,
                       device_train=device_train)

    ppo.learn()


def test():
    env = gym.make('CartPole-v1')

    policy = MyNetwork(in_dim=env.observation_space.shape[0], out_dim=env.action_space.n)

    policy.load_state_dict(torch.load('ppo_actor.pth'))

    for _ in range(100):
        state = env.reset()
        done = False

        traj_reward = 0
        while not done:
            logits = policy(torch.tensor(state))
            pdf = Categorical(logits=logits)
            action = pdf.sample().item()

            state, reward, done, info = env.step(action)

            traj_reward += reward
            env.render()

        logging.info(f'Trajectory reward:{traj_reward}')


def get_device():
    if torch.cuda.is_available():
        return torch.device("cpu"), torch.device("cuda")
    else:
        try:
            # For apple silicon
            if torch.backends.mps.is_available():
                return torch.device("cpu"), torch.device("mps")
            else:
                return torch.device("cpu"), torch.device("cpu")
        except Exception as e:
            logging.error(e)
            return torch.device("cpu"), torch.device("cpu")


if __name__ == '__main__':
    wandb.init(project='test-project-2', entity='point-goal-navigation', name=str(datetime.now()))

    # device_infer is used by collector to get outputs and device_train is used to train the network
    device_infer, device_train = get_device()

    train()
    # test()

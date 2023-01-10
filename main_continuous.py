import logging
import os
from datetime import datetime

import gym
import torch

import wandb
from network_continuous import ActorNetwork, CriticNetwork
from ppo_continuous import PPO_Continuous


def train():
    env_name = 'BipedalWalker-v3'

    env = gym.make(env_name)

    hyper_params = dict(
        n_epochs=200_000_000,
        # Total number of steps taken that follows training the network
        n_timesteps=4000,
        # Maximum number of steps taken in the environment if it does not terminate
        max_trajectory_length=500,
        # Total number of training steps at each epoch
        n_training_per_epoch=80,
        # Learning rate for actor
        actor_lr=3e-4,
        # Learning rate for critic
        critic_lr=1e-3,
        # Discount factor
        gamma=0.99,
        # Surrogate clip for actor loss
        clip=0.2,
        # KL above threshold terminates the learning
        kl_threshold=float('inf'),
        # Entropy penalty to encourage exploration and prevent policy from becoming too deterministic
        entropy_coef=0.001,
        # Timestep at which environment should render. There will be n_"timesteps / render_freq" total renders
        render_freq=0,
        # Number of workers that collects data parallely. Each will collect "n_timesteps / n_worker" timesteps
        n_workers=8
    )

    ppo = PPO_Continuous(training_name=env_name,
                         actor=ActorNetwork(in_dim=env.observation_space.shape[0], out_dim=env.action_space.shape[0]),
                         critic=CriticNetwork(in_dim=env.observation_space.shape[0], out_dim=1),
                         env=env,
                         device_infer=device_infer,
                         device_train=device_train,
                         hyper_params=hyper_params)

    ppo.learn()


def test():
    env = gym.make('BipedalWalker-v3')

    policy = ActorNetwork(in_dim=env.observation_space.shape[0], out_dim=env.action_space.shape[0])

    policy.load_state_dict(torch.load('saved_models/ppo_actor-BipedalWalker-v3.pth', map_location=device_infer))

    for _ in range(100):
        state = env.reset()
        done = False

        traj_reward = 0
        while not done:
            pdf = PPO_Continuous._get_action_pdf(policy, torch.tensor(state))

            action = pdf.sample().numpy()

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
    # Edit entries here
    wandb.init(project='BipedalWalker-v3', entity='point-goal-navigation', name=str(datetime.now()))

    os.makedirs('saved_models', exist_ok=True)

    # device_infer is used by collector to get outputs and device_train is used to train the network
    device_infer, device_train = get_device()

    train()
    # test()

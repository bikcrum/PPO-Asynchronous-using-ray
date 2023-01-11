import logging
import os

import gym
import torch

from network import ActorNetworkContinuous, CriticNetworkContinuous, NetworkDiscrete
from ppo import PPO, Utils


def train(env_name):
    os.makedirs('saved_models', exist_ok=True)

    env = gym.make(env_name)

    # HYPER PARAMETERS
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
        entropy_coef=0.02,
        # Timestep at which environment should render. There will be n_"timesteps / render_freq" total renders
        render_freq=0,
        # Number of workers that collects data parallely. Each will collect "n_timesteps / n_worker" timesteps
        n_workers=16
    )

    if is_continuous:
        actor = ActorNetworkContinuous(in_dim=env.observation_space.shape[0], out_dim=env.action_space.shape[0])
        critic = CriticNetworkContinuous(in_dim=env.observation_space.shape[0], out_dim=1)
    else:
        actor = NetworkDiscrete(in_dim=env.observation_space.shape[0], out_dim=env.action_space.n)
        critic = NetworkDiscrete(in_dim=env.observation_space.shape[0], out_dim=1)

    ppo = PPO(training_name=env_name,
              actor=actor,
              critic=critic,
              env=env,
              is_continuous=is_continuous,
              device_infer=device_infer,
              device_train=device_train,
              hyper_params=hyper_params)

    ppo.learn()


def test(env_name):
    env = gym.make(env_name)

    if is_continuous:
        policy = ActorNetworkContinuous(in_dim=env.observation_space.shape[0], out_dim=env.action_space.shape[0])
    else:
        policy = NetworkDiscrete(in_dim=env.observation_space.shape[0], out_dim=env.action_space.n)

    policy.load_state_dict(torch.load(f'saved_models/ppo_actor-{env_name}.pth', map_location=device_infer))

    for _ in range(100):
        state = env.reset()
        done = False

        traj_reward = 0
        while not done:
            action = Utils.get_action(policy, state, is_continuous=is_continuous).numpy()

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
    # device_infer is used by collector to get outputs and device_train is used to train the network
    device_infer, device_train = get_device()

    # Edit entries here
    # env_name, is_continuous = 'Acrobot-v1', False
    # env_name, is_continuous = 'BipedalWalker-v3', True
    # env_name, is_continuous = 'CartPole-v0', False
    # env_name, is_continuous = 'HalfCheetah-v2', True
    # env_name, is_continuous = 'MountainCar-v0', False
    # env_name, is_continuous = 'MountainCarContinuous-v0', True
    env_name, is_continuous = 'Pendulum-v1', True
    train(env_name)
    # test(env_name)

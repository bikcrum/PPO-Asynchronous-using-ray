import itertools
import logging
import time
from datetime import datetime

import numpy as np
import ray
import torch
import torch.nn as nn
from torch.distributions import kl_divergence
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Categorical
from torch.optim import Adam

import wandb

ray.init(num_cpus=20, ignore_reinit_error=True)
# ray.init(local_mode=True)

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

debug = False


class Utils:
    @staticmethod
    def merge_experiences(experiences):
        out = list(zip(*ray.get(experiences)))

        return [torch.concatenate(x) for x in out[:-2]], [
            list(itertools.chain(*x)) for x in out[-2:]]

    @staticmethod
    def compute_returns(terminal_value, ep_reward, gamma):
        ep_returns = []

        discounted_reward = terminal_value

        for rew in ep_reward[::-1]:
            discounted_reward = rew + discounted_reward * gamma
            ep_returns.append(discounted_reward)

        return ep_returns[::-1]

    @staticmethod
    def get_action_pdf(policy, states, is_continuous):
        if is_continuous:
            return Utils._get_continuous_action_pdf(policy, states)
        return Utils._get_discrete_action_pdf(policy, states)

    @staticmethod
    def get_action(policy, state, is_continuous):
        if is_continuous:
            return Utils._get_continuous_action(policy, state)
        return Utils._get_discrete_action(policy, state)

    @staticmethod
    def _get_continuous_action_pdf(policy, states):
        mu, sigma = policy(states)

        pdf = MultivariateNormal(mu, torch.diag_embed(sigma))
        return pdf

    @staticmethod
    def _get_discrete_action_pdf(policy, states):
        logits = policy(states)
        pdf = Categorical(logits=logits)
        return pdf

    @staticmethod
    def _get_continuous_action(actor, state):
        pdf = Utils._get_continuous_action_pdf(actor, state)
        return pdf.sample()

    @staticmethod
    def _get_discrete_action(actor, state):
        pdf = Utils._get_discrete_action_pdf(actor, state)
        return pdf.sample()


@ray.remote
class ExperienceReplay:
    def __init__(self,
                 env,
                 gamma,
                 n_timesteps,
                 max_trajectory_length,
                 render_freq,
                 device,
                 n_workers):
        self.env = env
        self.gamma = gamma
        self.n_timesteps = n_timesteps
        self.max_trajectory_length = max_trajectory_length
        self.render_freq = render_freq
        self.device = device
        self.n_workers = n_workers

    def rollout(self, actor, critic, is_continuous):
        states, actions, rewards, values, returns, lengths = [], [], [], [], [], []

        curr_timesteps = 0

        with torch.no_grad():
            while curr_timesteps < self.n_timesteps // self.n_workers:
                ep_rewards = []
                state = self.env.reset()
                done = False

                for curr_traj in range(self.max_trajectory_length):
                    curr_timesteps += 1

                    state = torch.tensor(state, device=self.device, dtype=torch.float)
                    states.append(state)

                    action = Utils.get_action(actor, state, is_continuous=is_continuous)

                    value = critic(state)

                    state, reward, done, _ = self.env.step(action.numpy())

                    if self.render_freq > 0 and curr_timesteps % self.render_freq == 0:
                        self.env.render()

                    actions.append(action)
                    values.append(value)

                    ep_rewards.append(reward)

                    if done:
                        break

                value = 0 if done else critic(torch.tensor(state, device=self.device, dtype=torch.float))
                ep_returns = Utils.compute_returns(terminal_value=value, ep_reward=ep_rewards, gamma=self.gamma)

                lengths.append(len(ep_rewards))
                rewards.append(ep_rewards)
                returns += ep_returns

        states = torch.stack(states).squeeze()
        actions = torch.stack(actions)
        values = torch.stack(values).squeeze()
        returns = torch.tensor(returns, dtype=torch.float32).squeeze()

        return states, actions, values, returns, lengths, rewards


class PPO:
    def __init__(self, training_name,
                 actor, critic, env, is_continuous, device_infer=torch.device('cpu'),
                 device_train=torch.device('cpu'),
                 hyper_params=None):

        # HYPER PARAMETERS
        self.n_epochs = hyper_params['n_epochs']
        self.n_timesteps = hyper_params['n_timesteps']
        self.max_trajectory_length = hyper_params['max_trajectory_length']
        self.n_training_per_epoch = hyper_params['n_training_per_epoch']
        self.actor_lr = hyper_params['actor_lr']
        self.critic_lr = hyper_params['critic_lr']
        self.gamma = hyper_params['gamma']
        self.clip = hyper_params['clip']
        self.kl_threshold = hyper_params['kl_threshold']
        self.entropy_coef = hyper_params['entropy_coef']
        self.render_freq = hyper_params['render_freq']
        self.n_workers = hyper_params['n_workers']

        self.is_continuous = is_continuous

        self.training_name = training_name
        self.device_infer = device_infer
        self.device_train = device_train

        self.env = env

        self.actor = actor
        self.critic = critic

        self.actor_optim = Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.critic_lr)

        self.workers = [ExperienceReplay.remote(env=self.env,
                                                gamma=self.gamma,
                                                n_timesteps=self.n_timesteps,
                                                max_trajectory_length=self.max_trajectory_length,
                                                render_freq=self.render_freq,
                                                device=self.device_infer,
                                                n_workers=self.n_workers) for _ in range(self.n_workers)]

    def learn(self):
        if not debug:
            wandb.init(project=self.training_name, entity='point-goal-navigation', name=str(datetime.now()))

        curr_epochs = 0
        max_reward = float('-inf')
        learning_complete = False

        start = time.time()

        while not learning_complete and curr_epochs < self.n_epochs:
            logging.info(f'EPOCH:{curr_epochs} Collecting experience')

            actor = self.actor.to(self.device_infer)
            critic = self.critic.to(self.device_infer)

            experiences = [
                experience_replay.rollout.remote(actor=actor,
                                                 critic=critic,
                                                 is_continuous=self.is_continuous) for experience_replay in self.workers
            ]

            (states, actions, values, returns), (lengths, rewards) = Utils.merge_experiences(experiences)

            logging.info('Training network')

            self.actor = self.actor.to(self.device_train)
            self.critic = self.critic.to(self.device_train)
            states = states.to(self.device_train)
            actions = actions.to(self.device_train)
            values = values.to(self.device_train)
            returns = returns.to(self.device_train)

            curr_epochs += np.sum(lengths)

            actor_losses = []
            critic_losses = []
            kls = []

            with torch.no_grad():
                pdf = Utils.get_action_pdf(self.actor, states, is_continuous=self.is_continuous)
                log_probs = pdf.log_prob(actions)

            advantage = returns - values
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)

            for _ in range(self.n_training_per_epoch):
                curr_pdf = Utils.get_action_pdf(self.actor, states, is_continuous=self.is_continuous)
                curr_log_probs = curr_pdf.log_prob(actions)

                ratios = torch.exp(curr_log_probs - log_probs)
                surr1 = ratios * advantage
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantage
                entropy_penalty = -(self.entropy_coef * curr_pdf.entropy()).mean()

                actor_loss = (-torch.min(surr1, surr2)).mean()

                critic_loss = nn.MSELoss()(self.critic(states).squeeze(), returns)

                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()

                (actor_loss + entropy_penalty).backward(retain_graph=True)
                critic_loss.backward()

                self.actor_optim.step()
                self.critic_optim.step()

                actor_losses.append((actor_loss + entropy_penalty).item())
                critic_losses.append(critic_loss.item())

                with torch.no_grad():
                    kls.append(kl_divergence(pdf, curr_pdf).mean().item())
                    if kls[-1] > self.kl_threshold:
                        logging.info(f"KL threshold reached. {kls[-1]}")
                        learning_complete = True
                        break

            reward = np.mean([np.sum(ep_rewards) for ep_rewards in rewards])
            if not debug:
                wandb.log({'reward': reward,
                           'episode length': np.mean(lengths),
                           'actor loss': np.mean(actor_losses),
                           'critic loss': np.mean(critic_losses),
                           'time steps': curr_epochs,
                           'time elapsed': time.time() - start,
                           'kl divergence': np.mean(kls)})

            if reward > max_reward:
                torch.save(self.actor.state_dict(), f'saved_models/ppo_actor-{self.training_name}.pth')
                torch.save(self.critic.state_dict(), f'saved_models/ppo_critic-{self.training_name}.pth')
                max_reward = reward

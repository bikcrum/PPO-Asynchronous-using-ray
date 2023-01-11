import itertools
import logging
import time

import numpy as np
import ray
import torch
import torch.nn as nn
from torch.distributions import kl_divergence
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.optim import Adam

import wandb

ray.init(num_cpus=20, ignore_reinit_error=True)
# ray.init(local_mode=True)
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


class PPO_Continuous:
    def __init__(self, training_name,
                 actor, critic, env, device_infer=torch.device('cpu'),
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

        self.training_name = training_name
        self.device_infer = device_infer
        self.device_train = device_train

        self.env = env

        self.actor = actor
        self.critic = critic

        self.actor_optim = Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.critic_lr)

    def learn(self):
        curr_epochs = 0
        max_reward = float('-inf')
        learning_complete = False

        start = time.time()

        while not learning_complete and curr_epochs < self.n_epochs:
            logging.info(f'EPOCH:{curr_epochs} Collecting experience')
            (states, actions, values, returns), (lengths, rewards)= PPO_Continuous._rollout(
                actor=self.actor,
                critic=self.critic,
                env=self.env,
                gamma=self.gamma,
                n_timesteps=self.n_timesteps,
                max_trajectory_length=self.max_trajectory_length,
                render_freq=self.render_freq,
                device=self.device_infer,
                n_workers=self.n_workers)

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
                pdf = self._get_action_pdf(self.actor, states)
                log_probs = pdf.log_prob(actions)

            advantage = returns - values
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)

            for _ in range(self.n_training_per_epoch):
                curr_pdf = PPO_Continuous._get_action_pdf(self.actor, states)
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

    @staticmethod
    def _rollout(actor, critic, env, gamma, n_timesteps, max_trajectory_length, render_freq, device, n_workers):
        actor = actor.to(device)
        critic = critic.to(device)

        def get_action(state):
            pdf = PPO_Continuous._get_action_pdf(actor, state)
            return pdf.sample()

        @ray.remote
        def worker(i):
            states, actions, rewards, values, returns, lengths = [], [], [], [], [], []

            curr_timesteps = 0

            with torch.no_grad():
                while curr_timesteps < n_timesteps // n_workers:
                    ep_rewards = []
                    state = env.reset()
                    done = False

                    for curr_traj in range(max_trajectory_length):
                        curr_timesteps += 1

                        state = torch.tensor(state, device=device, dtype=torch.float)
                        states.append(state)

                        action = get_action(state)

                        value = critic(state)

                        state, reward, done, _ = env.step(action.numpy())

                        if render_freq > 0 and curr_timesteps % render_freq == 0:
                            env.render()

                        actions.append(action)
                        values.append(value)

                        ep_rewards.append(reward)

                        if done:
                            break

                    value = 0 if done else critic(torch.tensor(state, device=device, dtype=torch.float))
                    ep_returns = PPO_Continuous._compute_returns(terminal_value=value, ep_reward=ep_rewards,
                                                                 gamma=gamma)

                    lengths.append(len(ep_rewards))
                    rewards.append(ep_rewards)
                    returns += ep_returns

            states = torch.stack(states).squeeze()
            actions = torch.stack(actions)
            values = torch.stack(values).squeeze()
            returns = torch.tensor(returns, dtype=torch.float32).squeeze()

            return states, actions, values, returns, lengths, rewards

        # return worker(0)

        out = list(zip(*ray.get([worker.remote(i) for i in range(n_workers)])))

        return [torch.concatenate(x) for x in out[:-2]], [list(itertools.chain(*x)) for x in out[-2:]]

    @staticmethod
    def _compute_returns(terminal_value, ep_reward, gamma):
        ep_returns = []

        discounted_reward = terminal_value

        for rew in ep_reward[::-1]:
            discounted_reward = rew + discounted_reward * gamma
            ep_returns.append(discounted_reward)

        return ep_returns[::-1]

    @staticmethod
    def _get_action_pdf(policy, states):
        mu, sigma = policy(states)

        pdf = MultivariateNormal(mu, torch.diag_embed(sigma))
        return pdf

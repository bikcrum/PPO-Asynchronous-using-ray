import itertools
import logging

import time

import numpy as np
import ray
import torch
import torch.nn as nn
from torch.distributions import kl_divergence
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import wandb

ray.init(num_cpus=20, ignore_reinit_error=True)
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


class PPO_Discrete:
    def __init__(self, policy_class, env, device_infer=torch.device('cpu'), device_train=torch.device('cpu')):
        # HYPER PARAMETERS

        # Total number of iterations each containing experience collection and learning
        self.n_epochs = 200_000_000
        # Total number of steps taken in the environment including termination between each
        self.n_timesteps = 2048
        # Maximum number of steps taken in the environment if it does not terminate
        self.max_trajectory_length = 200
        # Total number of training steps at each epoch
        self.n_training_per_epoch = 10
        # Learning rate
        self.lr = 3e-4
        # Discount factor
        self.gamma = 0.99
        # Surrogate clip for actor loss
        self.clip = 0.2
        # KL above threshold terminates the learning
        self.kl_threshold = 0.02
        # Entropy penalty to encourage exploration and prevent policy from becoming too deterministic
        self.entropy_coef = 0.01
        # Timestep at which environment should render. There will be n_"timesteps / render_freq" total renders
        self.render_freq = 0
        # Number of workers that collects data parallely. Each will collect "n_timesteps / n_worker" timesteps
        self.n_workers = 4

        self.device_infer = device_infer
        self.device_train = device_train

        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n

        self.policy_class = policy_class
        self.actor = policy_class(self.obs_dim, self.act_dim)
        self.critic = policy_class(self.obs_dim, 1)

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

    def learn(self):
        curr_epochs = 0
        max_reward = float('-inf')
        learning_complete = False

        start = time.time()

        while not learning_complete and curr_epochs < self.n_epochs:
            logging.info(f'EPOCH:{curr_epochs} Collecting experience')
            (states, actions, values, returns), (lengths, rewards) = self._rollout(
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
                curr_pdf = self._get_action_pdf(self.actor, states)
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

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())

                with torch.no_grad():
                    kls.append(kl_divergence(pdf, curr_pdf).mean().item())
                    if kls[-1] > self.kl_threshold:
                        learning_complete = True
                        break

            reward = np.mean([np.sum(ep_rewards) for ep_rewards in rewards])
            wandb.log({'reward': reward,
                       'episode length': np.mean([np.sum(ep_rewards) for ep_rewards in rewards]),
                       'actor loss': np.mean(actor_losses),
                       'critic loss': np.mean(critic_losses),
                       'time steps': curr_epochs,
                       'time elapsed': time.time() - start,
                       'kl divergence': np.mean(kls)})

            if reward > max_reward:
                torch.save(self.actor.state_dict(), 'ppo_actor.pth')
                torch.save(self.critic.state_dict(), 'ppo_critic.pth')
                max_reward = reward

    @staticmethod
    def _rollout(actor, critic, env, gamma, n_timesteps, max_trajectory_length, render_freq, device, n_workers):
        actor = actor.to(device)
        critic = critic.to(device)

        def get_action(state):
            pdf = PPO_Discrete._get_action_pdf(actor, state)
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

                        state = torch.tensor(state, device=device)
                        states.append(state)

                        action = get_action(state)

                        value = critic(state)

                        state, reward, done, _ = env.step(action.item())

                        if render_freq > 0 and curr_timesteps % render_freq == 0:
                            env.render()

                        actions.append(action)
                        values.append(value)

                        ep_rewards.append(reward)

                        if done:
                            break

                    value = 0 if done else critic(torch.tensor(state))
                    ep_returns = PPO_Discrete._compute_returns(terminal_value=value, ep_reward=ep_rewards, gamma=gamma)

                    lengths.append(len(ep_rewards))
                    rewards.append(ep_rewards)
                    returns += ep_returns

            states = torch.stack(states)
            actions = torch.tensor(actions)
            values = torch.tensor(values)
            returns = torch.tensor(returns)

            return states, actions, values, returns, lengths, rewards

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
        logits = policy(states)
        pdf = Categorical(logits=logits)
        return pdf

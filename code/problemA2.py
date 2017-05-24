import numpy as np
import pandas as pd
import gym
from gym.envs.registration import register

AGENT_CARTPOLE = 'CartPole-v0'
AGENT_CARTPOLE_300 = 'CartPole-300-v0'
AGENT_PACMAN = 'MsPacman-v0'
AGENT_PONG = 'Pong-v0'
AGENT_BOXING = 'Boxing-v0'

register(
    id='CartPole-300-v0',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 300},
    reward_threshold=1000.0,
)



def computeReturn(episodes_, episodes_lens_, damping_):
    G = np.sum(list(map(lambda i, x: (damping_ ** i) * x, episodes_lens_, episodes_)), axis=1)
    returns_dict = {'Mean(Episode lengths)': [np.mean(episodes_lens_)], 'SD(Episode lengths)': [np.std(episodes_lens_)],
                    'Mean(Return(G))': [np.mean(G)], 'SD(Return(G))': [np.std(G)]}
    pd.DataFrame(returns_dict).to_csv('../outputs/problemA2.csv', index=False)
    print('Mean(Episode lengths), SD(Episode lengths), Mean(Return(G)), SD(Return(G))')
    print(np.mean(episodes_lens_), np.std(episodes_lens_), np.mean(G), np.std(G))
    return G


def main(agent):
    MAX_EPISODE_LEN = 300
    N_EPISODES = 100

    DAMPING_FACTOR = 0.99
    episodes = np.zeros(shape=[N_EPISODES, MAX_EPISODE_LEN], dtype=int)
    episode_lens = np.zeros(shape=[N_EPISODES], dtype=int)

    env = gym.make(agent)

    for i_episode in range(N_EPISODES):

        observation = env.reset()
        currentEpisode = episodes[i_episode, :]

        for t in range(MAX_EPISODE_LEN):
            env.render()
            # print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            # print(reward)

            if done:
                currentEpisode[t] = -1
                episode_lens[i_episode] = t + 1
                # print('Episode finished after {} timesteps'.format(t + 1))
                break

    computeReturn(episodes, episode_lens, DAMPING_FACTOR)


if __name__ == '__main__':
    main(AGENT_CARTPOLE_300)

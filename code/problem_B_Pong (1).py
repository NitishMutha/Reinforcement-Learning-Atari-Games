# Course: COMPGI13- Advanced Topics in Machine Learning
# Institution: University College London
# Developer: Russel Daries (16079408)
# Question: B - Game of Pong
# Purpose: Implementing Epsilon-Greedy Q-Learning with Experience Replay Buffer for Pong

# Add additional directories
import sys
# Directory for common function files
sys.path.insert(0, '../common')

# Nesscary Import packages
import tensorflow as tf
import numpy as np
import pandas as pd
import gym
import random
import sys
import matplotlib
import matplotlib.pyplot

# Import to disable Windows X backend to be able to create plots in SSH session
matplotlib.use('Agg')

from collections import deque
from gym.envs.registration import register

# Local imports for common files
from class_definitions import *
from misc_functions import *

# Decalrations of various enviroment for Part A

ENV_CARTPOLE = 'CartPole-v0'
ENV_CARTPOLE_300 = 'CartPole-300-v0'

# Declarations of various enviroment for Part B

ENV_PONG = 'Pong-v3'
ENV_PACMAN = 'MsPacman-v3'
ENV_BOXING = 'Boxing-v3'

# # Set Register enviroment variables
#
# register(id='CartPole-300-v0',entry_point='gym.envs.classic_control:CartPoleEnv',
#          tags={'wrapper_config.TimeLimit.max_episode_steps': 300},
#          reward_threshold = 1000.0)

# Boolean and condition statement flags
# Model type
BRAIN_TYPE = 'Non-Linear'
OPTIMIZER = 'RMS'
TRAIN_MODE = True
RELOAD_MODEL = False
SAVE_PATH_VARIABLES = True

# Parameter setting
# LEARNING_RATES = [0.00001,0.0001,0.001,0.01,0.1,0.5]
LEARNING_RATES = [0.001]
EPOCHS = 3
BATCH_SIZE = 32
HIDDEN_NEURONS = 100
NUM_EPISODES = 10000
SEEDING = 200
EXPERIENCE_BUFFER_SIZE = 5000
CONSTRUCT_AGENT = False
OPERATION = 1
DISCOUNT_FACTOR = 0.99

# Problem Number
problem_num = 'B_Pong'

# Problem Model Path
model_path = '../../models/models/problem_'+str(problem_num)+'/'+BRAIN_TYPE+'/'
# Problem Variable Path
variable_path = '../../models/models/problem_'+str(problem_num)+'/'+BRAIN_TYPE+'/'
# Problem Plot Path
plot_path = '../../models/outputs/plots/problem_'+str(problem_num)+'/'+BRAIN_TYPE+'/'
# Problem Table Path
table_path = '../../models/outputs/tables/problem_'+str(problem_num)+'/'+BRAIN_TYPE+'/'
# Tensorflow Summary Path
tf_path = '../../models/summary/problem_'+str(problem_num)+'/'+BRAIN_TYPE+'/'

if(SAVE_PATH_VARIABLES):
    # All paths save directory
    all_save_path = '../../models/models/problem_'+str(problem_num)+'/'+BRAIN_TYPE+'/saved_paths.npz'
    np.savez(all_save_path,brain_type = BRAIN_TYPE, model_path=model_path,variable_path=variable_path,plot_path=plot_path,table_path=table_path,tf_path=tf_path)
    print('Variables saved to: '+ all_save_path)

def main(agent):

    np.random.seed(SEEDING)

    random.seed(SEEDING)

    env = gym.make(agent)

    agent = Agent(env, BRAIN_TYPE, LEARNING_RATES, SEEDING, OPTIMIZER)


    if(OPERATION==1):
        print('---Operation 1 occuring---')
        all_rewards = []
        all_frame_counts = []
        all_scores = []

        for episode_count in range(100):

            observe = env.reset()
            score = 0
            episode_reward = 0

            for i in range(env._max_episode_steps):

                if(CONSTRUCT_AGENT):
                    env.render()

                action = env.action_space.sample()

                observation,reward,done,_ = env.step(action)

                reward = agent.limit_reward(reward)
                score = score+reward
                episode_reward += (reward*(DISCOUNT_FACTOR**i))

                if(done):
                    all_scores.append(score)
                    all_frame_counts.append([i+1])
                    all_rewards.append(episode_reward)
                    break

        print('Mean Score (cumulative undiscounted rewards): ' + str(np.mean(all_rewards)) + ', Std Score: ' + str(np.std(all_rewards)))
        print('Mean Frame Counts: ' + str(np.mean(all_frame_counts)) + ', Std Frame Counts: ' + str(np.std(all_frame_counts)))
        print('Mean Score: ' + str(np.mean(all_scores)) + ', Std Score: ' + str(np.std(all_scores)))

    elif(OPERATION==2):
        print('---Operation 2 occuring---')

        with tf.Session() as sess:

            sess.run(agent.agent_brain.init)

            mean_reward, mean_frame, mean_score,std_reward, std_frame,std_score = agent.action(sess,100,CONSTRUCT_AGENT)

            print('Agent Mean Reward: ' + str(mean_reward) + ', Agent Std Reward: ' + str(std_reward) + ', Agent Mean Frame: ' + str(mean_frame) + ', Agent Std Frame: ' + str(std_frame))
            print('Agent Mean Score: ' + str(mean_score) + ', Agent Std Score: ' + str(std_score))

    elif(OPERATION==3):

        print('---Operation 3 occuring---')

        agent.replay(LEARNING_RATES,CONSTRUCT_AGENT,TRAIN_MODE,NUM_EPISODES,BATCH_SIZE,EPOCHS)
    else:
        pass

if __name__ =='__main__':

    main(ENV_PONG)
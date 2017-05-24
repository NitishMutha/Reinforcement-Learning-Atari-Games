# Course: COMPGI13- Advanced Topics in Machine Learning
# Institution: University College London
# Developer: Russel Daries (16079408)
# Purpose: Common class constructors for various algorithms

# Add additional directories
import sys
# Directory for common function files
sys.path.insert(0, '../common')

# Import nesscary packages

import numpy as np
import pandas as pd
import tensorflow as tf
import random
import scipy.misc as sci
from skimage.transform import resize
from skimage.color import rgb2gray

from collections import deque
from misc_functions import *

SEEDING = 200

# Restoring saved paths that are linked to main file
RESTORE = False

if(RESTORE):
    problem_num = 'B_Pong'
    brain_type = 'Non-Linear'
    all_save_path = '../../models/models/problem_'+str(problem_num)+'/'+brain_type+'/saved_paths.npz'
    all_save_path = np.load(all_save_path)

    # Restoring saved variables from np file
    model_path = str(all_save_path['model_path'])
    variable_path = str(all_save_path['variable_path'])
    plot_path = str(all_save_path['plot_path'])
    table_path = str(all_save_path['table_path'])
    tf_path = str(all_save_path['tf_path'])
    BRAIN_TYPE = str(all_save_path['brain_type'])

    # Tensorflow writer
    writer = tf.summary.FileWriter(tf_path)

# Defining weight variable function
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

# Defining bias variable function
def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

# Agent class for
class Agent:

    def __init__(self,enviroment, brain_type, LEARNING_RATE,SEEDING,OPTIMIZER):

        self.frame_stack = 4
        self.experience_buffer_size = 5000
        self.DISCOUNT_FACTOR = 0.99
        self.MAX_EPISODES = enviroment._max_episode_steps
        self.EPSILON = 0.1
        self.RESULT_DISPLAY_COUNT = 1000

        self.SAVE_BEST_MODEL = False
        self.BEST = 150

        self.brain_type = brain_type
        self.env = enviroment
        self.state_dims = 784
        self.num_actions = self.env.action_space.n
        self.agent_brain = Agent_Brain(brain_type,self.num_actions,self.state_dims,LEARNING_RATE,SEEDING,OPTIMIZER)
        self.memory = Agent_Memory(self.experience_buffer_size)
        self.steps = 0

    def limit_reward(self, reward):
        if reward > 0:
            return 1
        elif reward < 0:
            return -1
        else:
            return 0

    # Method for actions
    def action(self,sess,experiments,CONSTRUCT_AGENT):

        all_Rewards = []
        all_Experiment_Lengths = []
        all_Scores = []

        # For-loops for number for experiments
        for epi in range(experiments):

            test_observation = self.atari_preprocessor(self.env.reset())

            self.memory.add_frame_test(test_observation,self.frame_stack)

            # Stacking all four frame on one another
            state = self.memory.compile_frames_test()

            reward_for_episode = 0
            score = 0

            # For-Loops for episodes
            for i in range(self.MAX_EPISODES):

                if(CONSTRUCT_AGENT):
                    # Render the agent enviroment
                    self.env.render()

                state_reshaped = np.reshape(state,(1,-1))

                action,_ = self.agent_brain.q_prediction(sess,state_reshaped)

                test_observation_next,reward,done,_ = self.env.step(action[0])
                reward = self.limit_reward(reward)

                test_observation_next_pro = self.atari_preprocessor(test_observation_next)

                self.memory.add_frame_test(test_observation_next_pro)

                state = self.memory.compile_frames_test()

                reward_for_episode = reward_for_episode + (reward * self.DISCOUNT_FACTOR ** i)
                score = score + reward


                # Check to see if enviroment is finished
                if(done):

                    all_Rewards.append(reward_for_episode)
                    all_Experiment_Lengths.append([i+1])
                    all_Scores.append(score)
                    break

        # Return calculations from method call
        return np.mean(all_Rewards),np.mean(all_Experiment_Lengths),np.mean(all_Scores),np.std(all_Rewards),np.std(all_Experiment_Lengths),np.std(all_Scores)

    def atari_preprocessor(self,input):

        crop_start = 35
        crop_stop = 195

        input = input[crop_start:crop_stop]
        input = input[::2,::2,0]

        input = input[input == 144] = 0
        input = input[input == 109] = 0

        input = resize(input, (120, 120), mode='reflect')
        input = resize(input, (90, 90), mode='reflect')
        input = resize(input, (28, 28), mode='reflect', preserve_range=True)
        input = rgb2gray(input)

        input_conversion = input.astype(np.float)
        input_conversion = input_conversion.ravel()


        return input_conversion

    # Observe the enviroment
    def observe_env(self,sample,episode_end):

        self.steps +=1
        self.memory.add_episode(sample)

        if episode_end:

            self.memory.episode_lens = np.append(arr= self.memory.episode_lens,
            values = [self.steps]).flatten()
            self.steps = 0

    def create_experience_replay_buffer(self,sess,max_iter,CONSTRUCT_AGENT):

        for iterations in range(max_iter):

            observation = self.atari_preprocessor(self.env.reset())

            self.memory.add_frame_train(observation,self.frame_stack)

            state = self.memory.compile_frames_train()


            for m in range(self.MAX_EPISODES):

                if(CONSTRUCT_AGENT):
                    self.env.render()

                temp_state_reshape = np.reshape(state,(1,-1))

                action,_ = self.agent_brain.q_prediction(sess, temp_state_reshape)

                next_observation,reward,done,_ = self.env.step(action[0])

                reward = self.limit_reward(reward)

                next_observation_proc = self.atari_preprocessor(next_observation)

                self.memory.add_frame_train(next_observation_proc)

                next_state = self.memory.compile_frames_train()

                if done:
                    next_state = None

                self.memory.add_episode((state,action[0],reward,next_state))

                state = next_state

                length_temp = len(self.memory.experience_buffer_episodes)

                if(self.experience_buffer_size == length_temp):
                    return

                if(done):
                    break


    #Method call for replay capability
    def replay(self,LEARNING_RATE,CONSTRUCT_AGENT,TRAIN_MODE,NUM_EPISODES,BATCH,EPOCHS):

        with tf.Session() as sess:

            sess.run(self.agent_brain.init)

            writer.add_graph(sess.graph)

            performance = []
            performance.append(0)
            loss = []
            loss.append(0.99)
            scores = []
            rewards = []

            self.create_experience_replay_buffer(sess,self.experience_buffer_size,CONSTRUCT_AGENT)

            if(TRAIN_MODE):

                print('------ Training Mode underway-----')

                for i in range(NUM_EPISODES):

                    observation = self.atari_preprocessor(self.env.reset())

                    self.memory.add_frame_train(observation,self.frame_stack)

                    state = self.memory.compile_frames_train()

                    for m in range(self.MAX_EPISODES):

                        if(CONSTRUCT_AGENT):
                            self.env.render()

                        temp_state_reshape = np.reshape(state,(1,-1))

                        action,Q = self.agent_brain.q_prediction(sess,temp_state_reshape)

                        rand_number = np.random.rand(1)

                        if(self.EPSILON>rand_number):

                            action[0] = self.env.action_space.sample()

                        next_observation,reward,done,_= self.env.step(action[0])

                        reward = self.limit_reward(reward)

                        next_observation_proc = self.atari_preprocessor(next_observation)

                        self.memory.add_frame_train(next_observation_proc)

                        next_state = self.memory.compile_frames_train()

                        if done:
                            next_state = None

                        self.memory.add_episode((state,action[0],reward,next_state))

                        batch_data = self.memory.episode_samples(BATCH)

                        results_rewards = list(map(lambda x:x[2],batch_data))
                        results_actions = list(map(lambda x:x[1],batch_data))
                        results_states = list(map(lambda x:x[0],batch_data))

                        states_reshaped = np.reshape(results_states,(BATCH,-1))

                        Q_max,Q = self.agent_brain.q_prediction(sess,states_reshaped)

                        zeros_states = np.zeros(self.state_dims)

                        resultant_states  = list(map(lambda x: zeros_states if x[3] is None else x[3],batch_data))

                        resultant_states_reshaped = np.reshape(resultant_states,(BATCH,-1))

                        Q_max_next,Q_next = self.agent_brain.q_prediction(sess,resultant_states_reshaped)

                        Q_target = np.copy(Q)

                        Q_update = results_rewards + (self.DISCOUNT_FACTOR* np.amax(Q_next,1))

                        Q_update = list(map(lambda x, y: results_rewards if y is None else x, Q_update, resultant_states))

                        Q_target = list(map(
                            lambda a, q_old, q_new: np.array(
                                [q_new if a == 0 else q_old[0], q_new if a == 1 else q_old[1],
                                 q_new if a == 2 else q_old[2], q_new if a == 3 else q_old[3],
                                 q_new if a == 4 else q_old[4], q_new if a == 5 else q_old[5]]),
                            results_states, Q_target, Q_update))

                        temp_reshaped_results_states = np.reshape(resultant_states,(BATCH,-1))

                        _ , agent_loss, summary = self.agent_brain.train(sess, x_input=temp_reshaped_results_states,y_output=Q_target)

                        writer.add_summary(summary, i)

                        state = next_state

                        loss.append(agent_loss)

                        if(done):

                            break


                    if(i%self.RESULT_DISPLAY_COUNT==0):


                        agent_rewards,agent_performances,agent_score, std_rewards, std_performances,std_score = self.action(sess,2,CONSTRUCT_AGENT)

                        print('Episode: '+ str(int(i)) + ', Performance:' + str(agent_performances)+', Reward:'+str(agent_rewards),', Score:'+str(agent_score))

                        # Append calculations to data structure
                        performance.append(agent_performances)
                        scores.append(agent_score)
                        rewards.append(agent_rewards)


                        # Conditional statement to save best performing model
                        if(self.BEST < agent_score):

                            self.BEST = agent_score
                            self.SAVE_BEST_MODEL = True

                            save_path = self.agent_brain.saver.save(sess,model_path+'/'+str(LEARNING_RATE)+'_model.ckpt')

                    if (i % 2000 == 0 and i != 0):
                        print('--Updating Target Network--')
                        self.agent_brain.t_network_update(sess)


                # Plot for agent rewards
                plot_data(rewards,'Epochs','Avg Cumlative Reward','r','Agent Rewards , learning rate = ' + str(LEARNING_RATE),plot_path+'_rewards_'+problem_num+brain_type)

                # Plot for agent loss
                plot_data(loss,'Epochs','Loss','b','Bellman Loss,  learning rate= '+ str(LEARNING_RATE),plot_path+'_loss_'+problem_num+brain_type)

                # Plot for agent performances
                plot_data(performance,'Epochs','Frame Counts','g','Agent Performance , learning rate = '+str(LEARNING_RATE),plot_path+'_performance_'+problem_num+brain_type)

                plot_data(scores,'Epochs','Score','b','Training Score,  learning rate= '+ str(LEARNING_RATE),plot_path+'_loss_'+problem_num+brain_type)


                # Conditional statement to continue saving model until best model has been saved
                if(self.SAVE_BEST_MODEL==False):

                    # Continue saving model until the best model has been saved
                    save_path = self.agent_brain.saver.save(sess, model_path+'model.ckpt')

                    # Save numpy variables
                np.savez(variable_path + '/backed_up_save.npz',performance = performance, loss = loss, all_mean_rewards = all_mean_rewards, scores = scores, rewards = rewards)

            else:
                # Conditional statement to restore model
                print('------ Restoring and testing model-------')
                self.agent_brain.saver.restore(sess,model_path + 'model.ckpt')

                agent_rewards,agent_performances,agent_score,std_rewards,std_performances,std_score = self.action(sess,100,CONSTRUCT_AGENT)

                print('Avg Agent Reward: ' + str(agent_rewards),'STD:' + str(std_rewards))
                print('Avg Agent Performance: ' + str(agent_performances),'STD:' + str(std_performances))
                print('Avg Agent Score: '+ str(agent_score),'STD:' + str(std_score))

                # Create output dictionary
                output_dict = {'Avg Agent Reward':[agent_rewards],'Avg Agent Performance':[agent_performances],
                               'Avg Agent Score':[agent_score]}

                # Create panda dataframe to output to csv file
                pd.DataFrame(output_dict).to_csv(table_path+'/'+problem_num+'.csv')

class Agent_Brain:

    def __init__(self,brain_type,num_actions,state_dims,LEARNING_RATE,SEEDING=0,OPTIMIZER = 'RMS'):

        self.image_size = 28
        self.frame_stack = 4
        tf.reset_default_graph()
        tf.set_random_seed(SEEDING)
        self.t_net_alpha = 1

        self.state_arrays = tf.cast(tf.placeholder(shape=[None, state_dims*self.frame_stack],dtype=tf.uint8), dtype=np.float32)
        self.next_state_arrays = tf.cast(tf.placeholder(shape=[None, state_dims*self.frame_stack],dtype=tf.uint8), dtype=np.float32)

        with tf.variable_scope('Q_net'):
            self.Q = self.create_q_network(brain_type,num_actions,self.state_arrays)

        with tf.variable_scope('T_net'):
            self.t_network = tf.stop_gradient(self.create_q_network(brain_type,num_actions,self.next_state_arrays))

        self.predict = tf.argmax(self.Q,1)

        self.t_network_predict = tf.argmax(self.t_network,1)

        self.Q_next = tf.placeholder(shape=[None,num_actions],dtype=tf.float32)

        self.loss = tf.reduce_mean(tf.square(self.Q_next-self.Q))

        if(OPTIMIZER=='RMS'):
            self.optimize = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE,momentum=0.9).minimize(self.loss)
        elif(OPTIMIZER=='ADAM'):
            self.optimize = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(self.loss)
        else:
            pass

        Q_net_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = 'Q_net')
        T_net_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = 'T_net')

        for variable_source, variable_target in zip(Q_net_var,T_net_var):

            update_procedure = variable_target.assign_sub(self.t_net_alpha * (variable_target - variable_source))

            self.network_target_update.append(update_procedure)

        self.network_target_update = tf.group(*self.network_target_update)

        # Record loss in tensorflow plots to be used for tensorboard
        tf.summary.scalar("loss",self.loss)
        self.summary_merged = tf.summary.merge_all()

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()


    # Function to train network
    def train(self,sess,x_input,y_output):

        return sess.run([self.optimize,self.loss,self.summary_merged],
                        feed_dict = {self.state_arrays:x_input, self.Q_next:y_output})

    # Function call to predict next Q (return)
    def q_prediction(self,sess,x_input):

        return sess.run([self.predict,self.Q], feed_dict = {self.state_arrays:x_input})

    # Q Network function creation
    def create_q_network(self,brain_type,num_actions,state_dims):

        input_layer = tf.reshape(state_dims,[-1,self.image_size,self.image_size,self.frame_stack])

        conv_layer_1 = tf.layers.conv2d(inputs=input_layer,kernel_size=6,padding='valid',filters=16,strides=(2,2),activation=tf.nn.relu)
        # conv_layer_2 = tf.layers.conv2d(inputs=conv_layer_1,kernel_size=4,padding='valid',filters=32,strides=(2,2),activation=tf.nn.relu)

        conv_2d_flatten = tf.reshape(conv_layer_1,[-1,12 * 12 * 16])

        fully_connected = tf.layers.dense(inputs=conv_2d_flatten,units=256,activation=tf.nn.relu)

        Q = tf.layers.dense(inputs=fully_connected,units=num_actions)

        return Q


class Agent_Memory:


    def __init__(self,max_size):

        frame_stack = 4
        self.experience_buffer = deque(maxlen = max_size)
        self.size = max_size
        self.episode_lens = np.array([])

        # Buffer for frame stacking procedure

        self.frame_buffer_train = deque(maxlen = frame_stack)
        self.frame_buffer_test = deque(maxlen= frame_stack)

    def add_frame_train(self,frame,repeat = 1):

        for count in range(repeat):

            self.frame_buffer_train.append(frame)

    def add_frame_test(self, frame, repeat=1):

        for count in range(repeat):
            self.frame_buffer_test.append(frame)

    def compile_frames_train(self):

        compiled_frames_train = np.array(list(self.frame_buffer_train))

        return compiled_frames_train

    def compile_frames_test(self):

        compiled_frames_test = np.array(list(self.frame_buffer_test))

        return compiled_frames_test

    # Function to append each episode
    def add_episode(self,sample):

        self.experience_buffer.append(sample)

    # Function to sample under a random policy
    def episode_samples(self,batch):

        batch = min(batch,len(self.experience_buffer))
        return random.sample(tuple(self.experience_buffer),batch)
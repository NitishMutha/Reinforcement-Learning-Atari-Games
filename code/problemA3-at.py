import numpy as np
import tensorflow as tf
import gym
from gym.envs.registration import register
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
import random
import sys
import pandas as pd

AGENT_CARTPOLE = 'CartPole-v0'
AGENT_CARTPOLE_300 = 'CartPole-300-v0'
AGENT_PACMAN = 'MsPacman-v0'
AGENT_PONG = 'Pong-v0'
AGENT_BOXING = 'Boxing-v0'

LINEAR = True
RMS_OPTIMIZER = True

RENDER_AGENT = False
N_EPISODES = 2000
SEED = 0

BATCH = 10
EPOCH = 100
LEARNING_RATES = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5]
HIDDEN_LAYER = 100

RELOAD_MODEL = True
TRAIN_MODE = True
SAVE_PLOT = True
LOG_PATH = '../summary/problemA3'
PATH = ''

register(
    id='CartPole-300-v0',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 300},
    reward_threshold=1000.0,
)

writer = tf.summary.FileWriter(LOG_PATH)


class Brain:
    def __init__(self, liner_type, state_dim, n_actions, LEARNING_RATE):
        tf.reset_default_graph()
        tf.set_random_seed(SEED)

        # create computation graph
        self.state_vectors = tf.placeholder(shape=[None, state_dim], dtype=tf.float32, name='state_vectors')

        if liner_type:
            self.W = tf.Variable(tf.truncated_normal([state_dim, n_actions], stddev=0.01), dtype=np.float32,
                                 name='weights')
            # self.W = tf.get_variable(name='w1', shape=[state_dim, n_actions],
            #                          initializer=tf.contrib.layers.xavier_initializer(),
            #                          dtype=np.float32)
            self.b = tf.Variable(tf.truncated_normal([n_actions], stddev=0.01), name='bias')
            self.Q = tf.matmul(self.state_vectors, self.W) #+ self.b
            self.regularizer = tf.nn.l2_loss(self.W) #+ tf.nn.l2_loss(self.b)

        else:
            self.W1 = tf.Variable(tf.truncated_normal([state_dim, HIDDEN_LAYER], stddev=0.1), name='hidden_weight')
            self.b1 = tf.Variable(tf.zeros([HIDDEN_LAYER]), name='hidden_bias')
            self.h1 = tf.nn.relu(tf.matmul(self.state_vectors, self.W1) + self.b1, name='hidden_layer_out')

            self.W2 = tf.Variable(tf.truncated_normal([HIDDEN_LAYER, n_actions], stddev=0.1), name='output_weights')
            self.b2 = tf.Variable(tf.zeros([n_actions]), name='output_bias')
            self.Q = tf.matmul(self.h1, self.W2) + self.b2

            # self.regularizer = tf.nn.l2_loss(self.W1) + tf.nn.l2_loss(self.W2) + tf.nn.l2_loss(self.b2) + tf.nn.l2_loss(
            #     self.b2)

        self.predict = tf.argmax(self.Q, 1)

        self.next_Q = tf.placeholder(shape=[None, n_actions], dtype=tf.float32, name='next_q')

        self.loss = tf.reduce_mean(tf.square(self.next_Q - self.Q))
        #self.loss = tf.reduce_mean(self.loss_ + 0.01 * self.regularizer)

        if RMS_OPTIMIZER:
            self.optimize = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE, momentum=0.9,
                                                      name='trainer').minimize(
                self.loss)
        else:
            self.optimize = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, name='trainer').minimize(
                self.loss)
        tf.summary.scalar("loss", self.loss)
        self.merged_summary = tf.summary.merge_all()

        self.tf_init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()


    def predict_Q(self, sess, x):
        return sess.run([self.predict, self.Q], feed_dict={self.state_vectors: x})


    def train(self, sess, x, y):
        return sess.run([self.optimize, self.loss, self.merged_summary],
                        feed_dict={self.state_vectors: x, self.next_Q: y})


class Memory:
    def __init__(self, limit):
        self.episodes = deque(maxlen=limit)
        self.limit = limit
        self.episode_lens = np.array([])

    def add(self, sample):
        self.episodes.append(sample)

    def sample_episodes(self, batch):
        batch = min(batch, len(self.episodes))
        return random.sample(tuple(self.episodes), batch)


class Agent:
    MEMORY_LIMIT = 700000
    DAMPING_FACTOR = 0.99
    MAX_EPISODE_LEN = 300
    TOPMODEL_SAVED = False
    TOPNOTCH = 170

    def __init__(self, env, approximation_type, LEARNING_RATE):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.n_action = self.env.action_space.n
        self.brain = Brain(approximation_type, self.state_dim, self.n_action, LEARNING_RATE)
        self.memory = Memory(self.MEMORY_LIMIT)
        self.steps = 0

    # perform action
    def act(self, sess, iterations):
        allRewards = []
        allExperitmentLengths = []
        for itz in range(iterations):
            state = self.env.reset()
            for t in range(self.MAX_EPISODE_LEN):
                if RENDER_AGENT:
                    self.env.render()
                action, _ = self.brain.predict_Q(sess, np.expand_dims(state, axis=0))
                state, reward, done, _ = self.env.step(action[0])

                if done:
                    allRewards.append(-1 * self.DAMPING_FACTOR ** t if t < 300 else 0)
                    allExperitmentLengths.append([t + 1])
                    break
        return np.mean(allRewards), np.mean(allExperitmentLengths), np.std(allRewards), np.std(allExperitmentLengths)/2

    # save state
    def observe(self, sample, episode_end):

        self.steps += 1
        self.memory.add(sample)
        if episode_end:
            self.memory.episode_lens = np.append(arr=self.memory.episode_lens, values=[self.steps]).flatten()
            self.steps = 0

    # train
    def replay(self, LEARNING_RATE):

        with tf.Session() as sess:
            sess.run(self.brain.tf_init)
            writer.add_graph(sess.graph)

            performance = [0]
            all_mean_reward = []
            all_std_rewards = []
            all_std_performance = [0]
            loss = [0.6]

            if TRAIN_MODE:
                print('\n\n---Running in train mode---')
                for itr in range(EPOCH):

                    for ib in range(int(N_EPISODES / BATCH)):
                        batch = self.memory.sample_episodes(BATCH)
                        batchLen = len(batch)

                        # batch results
                        rewards = list(map(lambda x: x[2], batch))

                        # batch actions
                        actions = list(map(lambda x: x[1], batch))

                        # states = np.array([sample[0] for sample in batch], dtype=np.float32)
                        states = list(map(lambda x: x[0], batch))
                        max_Q, Q = self.brain.predict_Q(sess, states)

                        no_state = np.zeros(self.state_dim)
                        # resultant_states = np.array([(no_state if sample[3] is None else sample[3]) for sample in batch], dtype=np.float32)
                        resultant_states = list(map(lambda x: no_state if x[3] is None else x[3], batch))
                        max_next_Q, next_Q = self.brain.predict_Q(sess, resultant_states)

                        Q_target = np.copy(Q)
                        # targetQ[0, a[0]] = r + y * maxQ1
                        Q_update = rewards + self.DAMPING_FACTOR * np.amax(next_Q, 1)
                        Q_update = list(map(lambda x, y: y if y == -1 else x, Q_update, rewards))

                        Q_target = list(map(lambda a, q_old, q_new: np.array(
                                [q_new if a == 0 else q_old[0], q_new if a == 1 else q_old[1]]),
                            actions, Q_target, Q_update))

                        # train and compute gradients
                        _, agent_loss, summary = self.brain.train(sess, x=states, y=Q_target)
                        writer.add_summary(summary, itr)

                    # Run evaluation after each epoch
                    agent_reward, agent_performance, std_rewards, std_performance = self.act(sess, 100)

                    print('Epoch: ' + str(int(itr)) + ' Agent performance: ' + str(
                        agent_performance) + ' Loss: ' + str(
                        agent_loss))

                    performance.append(agent_performance)
                    all_mean_reward.append(agent_reward)
                    all_std_rewards.append(std_rewards)
                    all_std_performance.append(std_performance)
                    loss.append(agent_loss)

                    if agent_performance > self.TOPNOTCH:
                        self.TOPNOTCH = agent_performance
                        self.TOPMODEL_SAVED = True
                        save_path = self.brain.saver.save(sess, '../models/problemA3/' + PATH + '/' + str(
                            LEARNING_RATE) + '/model.ckpt')

                # plotGraph(data=performance[0:-1],
                #           title='Agent Performance (avg 100) - (learning rate ' + str(LEARNING_RATE) + ')',
                #           xlabel='Epochs', ylabel='Episode Length', label='Episode Length',
                #           sd=all_std_performance[0:-1])
                # plotGraph(data=all_mean_reward,
                #           title='Agent Rewards (avg 100) - (learning rate ' + str(LEARNING_RATE) + ')',
                #           xlabel='Epochs', ylabel='Rewards', label='rewards')
                # plotGraph(data=loss[0:-1], title='Agent Training Loss - (learning rate ' + str(LEARNING_RATE) + ')',
                #           xlabel='Epochs', ylabel='Loss', label='loss')

                plt.figure(np.random.randint(0,100), figsize=(14, 10), dpi=50)
                plt.subplot(311)
                plotGraph(data=performance[0:-1],
                          title='Agent Performance (avg 100) - (learning rate ' + str(LEARNING_RATE) + ')',
                          color='b', ylabel='Episode Length', label='Episode Length', sd=all_std_performance[0:-1])
                plt.subplot(312)
                plotGraph(data=all_mean_reward[0:-1],
                          title='Agent Rewards (avg 100) - (learning rate ' + str(LEARNING_RATE) + ')',
                          color='g', ylabel='Rewards', label='rewards')
                plt.subplot(313)
                plotGraph(data=loss[0:-1], title='Agent Training Loss - (learning rate ' + str(LEARNING_RATE) + ')',
                          color='r', ylabel='Bellman Loss', label='loss')
                plt.xlabel('Episodes', fontsize=18)
                if SAVE_PLOT:
                    plt.savefig('../plots/problemA3/' + PATH + '/plot-' + str(LEARNING_RATE) + '.eps', format='eps', dpi=50)
                else:
                    plt.show()

                if not self.TOPMODEL_SAVED:
                    save_path = self.brain.saver.save(sess, '../models/problemA3/' + PATH + '/' + str(
                        LEARNING_RATE) + '/model.ckpt')
                    print('Model weights saved in file: ', save_path)

                np.savez('../models/problemA3/' + PATH + '/' + str(LEARNING_RATE) + '/savedata.npz', loss=loss,
                         performance=performance)

            else:
                # test mode
                print('\n\n---Running in test mode---')
                self.brain.saver.restore(sess, '../models/problemA3/' + PATH + '/' + str(LEARNING_RATE) + '/model.ckpt')
                agent_reward, agent_performance, std_rewards, std_performance = self.act(sess, 100)
                print('Average Agent Reward: ' + str(
                    agent_reward) + ', Average Agent performance (episode length): ' + str(agent_performance))

                performance_dict = {'Avg Score (reward)': [agent_reward],
                                    'Avg episode Counts': [agent_performance]}
                pd.DataFrame(performance_dict).to_csv('../outputs/problemA3-Hidden' + str(LEARNING_RATE) + '.csv')

    # generate episodes training data
    def generate_experience(self, NUMBER_EXP):
        # create training dataset
        for i_episode in range(NUMBER_EXP):

            state = self.env.reset()

            for t in range(self.MAX_EPISODE_LEN):
                if RENDER_AGENT:
                    self.env.render()

                action = self.env.action_space.sample()
                next_state, reward, done, _ = self.env.step(action)

                if done:
                    next_state = None
                    reward = -1 if t < 300 else 0
                else:
                    reward = 0

                self.observe((state, action, reward, next_state, i_episode), done)

                state = next_state

                if done:
                    break


def plotGraph(data, label, color, ylabel, title, sd=None):

    plt.plot(data, color=color, label=label, lw=2)
    if sd is not None:
        x = np.arange(0, len(data), 1)
        plt.fill_between(x=x,y1=np.add(data,sd),y2=np.subtract(data,sd),alpha=0.1)
    plt.grid(True)
    #plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    #plt.title(title, fontsize=22, fontweight='bold')
    plt.legend()


def main(agent):
    np.random.seed(SEED)
    random.seed(a=SEED)

    env = gym.make(agent)

    if not RELOAD_MODEL:
        agent = Agent(env, LINEAR, 0.01)
        agent.generate_experience(N_EPISODES)
        np.savez('../models/problemA3/problemA3episodes.npz', memory=agent.memory.episodes,
                 episode_lens=agent.memory.episode_lens)
        print('episodes saved...')

    for lr in LEARNING_RATES:
        print('#Agent = Learning rate: ' + str(lr) + ' #Network= ' + 'Linear:' + str(LINEAR))
        agent = Agent(env, LINEAR, lr)
        saved_episodes = np.load('../models/problemA3/problemA3episodes.npz')
        agent.memory.episodes = saved_episodes['memory']
        agent.memory.episode_lens = saved_episodes['episode_lens']

        agent.replay(lr)


if __name__ == '__main__':
    args = sys.argv
    if len(args) > 1:
        TRAIN_MODE = args[1].lower() == 'train'
        if len(args) > 2:
            LINEAR = args[2].lower() == 'linear'
        if len(args) > 3:
            BATCH = int(args[3])
        if len(args) > 4:
            EPOCH = int(args[4])
    else:
        TRAIN_MODE = False  # todo update to False
        LINEAR = True

    if LINEAR:
        PATH = 'linear'
    else:
        PATH = 'hidden'
    main(AGENT_CARTPOLE_300)
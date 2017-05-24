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

LINEAR = False
RMS_OPTIMIZER = True

RENDER_AGENT = False
N_EPISODES = 2000
EXPERIENCE_BUFFER = 10000
SPIT_RESULTS = 20
SEED = 0

BATCH = 500
EPOCH = 60
LEARNING_RATES = [0.0001] #[0.0001, 0.0001, 0.001, 0.01, 0.1, 0.5]
HIDDEN_LAYER = 100

RELOAD_MODEL = True
TRAIN_MODE = True
SAVE_PLOT = True
LOG_PATH = '../summary/problemA7'

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
        self.target_update_rate = 1

        # create computation graph
        self.state_vectors = tf.placeholder(shape=[None, state_dim], dtype=tf.float32, name='state_vectors')
        self.next_state_vectors = tf.placeholder(shape=[None, state_dim], dtype=tf.float32, name='next_state_vectors')

        with tf.variable_scope('q_network'):
            self.Q = self.get_q_network(liner_type, self.state_vectors, state_dim, n_actions)

        with tf.variable_scope('target_network'):
            self.target_network = tf.stop_gradient(
                self.get_q_network(liner_type, self.next_state_vectors, state_dim, n_actions))

        self.predict = tf.argmax(self.Q, 1)
        self.predict_target_network = tf.argmax(self.target_network, 1)

        self.next_Q = tf.placeholder(shape=[None, n_actions], dtype=tf.float32, name='next_q')
        self.loss = tf.reduce_mean(tf.square(self.next_Q - self.Q))
        if RMS_OPTIMIZER:
            self.optimize = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE, momentum=0.9,
                                                      name='trainer').minimize(
                self.loss)
        else:
            self.optimize = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, name='trainer').minimize(
                self.loss)

        with tf.name_scope("update_target_network"):
            self.target_network_update = []
            # slowly update target network parameters with Q network parameters
            q_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="q_network")
            target_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target_network")
            for v_source, v_target in zip(q_network_variables, target_network_variables):
                # this is equivalent to target = (1-alpha) * target + alpha * source
                update_op = v_target.assign_sub(self.target_update_rate * (v_target - v_source))
                self.target_network_update.append(update_op)
            self.target_network_update = tf.group(*self.target_network_update)

        tf.summary.scalar("loss", self.loss)
        self.merged_summary = tf.summary.merge_all()

        self.tf_init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

    def get_q_network(self, liner_type, state, state_dim, n_actions):

        if liner_type:
            W = tf.Variable(tf.truncated_normal([state_dim, n_actions], stddev=0.1), dtype=np.float32,
                                 name='weights')
            Q = tf.matmul(state, W)
        else:
            W1 = tf.Variable(tf.truncated_normal([state_dim, HIDDEN_LAYER], stddev=0.1), name='hidden_weight')
            b1 = tf.Variable(tf.zeros([HIDDEN_LAYER]), name='hidden_bias')
            h1 = tf.nn.relu(tf.matmul(state, W1) + b1, name='hidden_layer_out')

            W2 = tf.Variable(tf.truncated_normal([HIDDEN_LAYER, n_actions], stddev=0.1), name='output_weights')
            b2 = tf.Variable(tf.zeros([n_actions]), name='output_bias')
            Q = tf.matmul(h1, W2) + b2

        return Q

    def predict_Q(self, sess, x):
        return sess.run([self.predict, self.Q], feed_dict={self.state_vectors: x})

    def predict_traget_Q(self, sess, x):
        return sess.run([self.predict_target_network, self.target_network], feed_dict={self.next_state_vectors: x})

    def train(self, sess, x, y):
        return sess.run([self.optimize, self.loss, self.merged_summary],
                        feed_dict={self.state_vectors: x, self.next_Q: y})

    def update_target_network(self, sess):
        return sess.run([self.target_network_update])


class Memory:
    def __init__(self, limit):
        self.experienceBuffer = deque(maxlen=limit)
        self.limit = limit
        self.episode_lens = np.array([])

    def add(self, sample):
        self.experienceBuffer.append(sample)

    def sample_episodes(self, batch):
        batch = min(batch, len(self.experienceBuffer))
        return random.sample(tuple(self.experienceBuffer), batch)


class Agent:
    DAMPING_FACTOR = 0.99
    MAX_EPISODE_LEN = 300
    EPSILON = 0.05
    TOPMODEL_SAVED = False
    TOPNOTCH = 250

    def __init__(self, env, approximation_type, LEARNING_RATE):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.n_action = self.env.action_space.n
        self.brain = Brain(approximation_type, self.state_dim, self.n_action, LEARNING_RATE)
        self.memory = Memory(EXPERIENCE_BUFFER)
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
    def createBuffer(self, sess, maxIterations):

        for itz in range(maxIterations):
            state = self.env.reset()
            for t in range(self.MAX_EPISODE_LEN):
                if RENDER_AGENT:
                    self.env.render()
                action, _ = self.brain.predict_Q(sess, np.expand_dims(state, axis=0))
                next_state, reward, done, _ = self.env.step(action[0])

                if done:
                    next_state = np.zeros(self.state_dim)
                    reward = -1 if t < 300 else 0
                else:
                    reward = 0

                self.memory.add((state, action[0], reward, next_state))
                state = next_state
                if len(self.memory.experienceBuffer) == EXPERIENCE_BUFFER:
                    return
                if done:
                    break

    # train
    def replay(self, LEARNING_RATE):

        with tf.Session() as sess:
            sess.run(self.brain.tf_init)
            writer.add_graph(sess.graph)

            # create a buffer
            self.createBuffer(sess, EXPERIENCE_BUFFER)

            performance = [0]
            all_mean_reward = [-1]
            all_std_rewards = []
            all_std_performance = [0]
            loss = [0.99]
            if TRAIN_MODE:
                print('\n\n---Running in train mode---')
                for itr in range(N_EPISODES):

                    state = self.env.reset()

                    for t in range(self.MAX_EPISODE_LEN):
                        if RENDER_AGENT:
                            self.env.render()
                        action, Q = self.brain.predict_Q(sess, np.expand_dims(state, axis=0))

                        # Exploration by e-greedy
                        if np.random.rand(1) < self.EPSILON:
                            action[0] = self.env.action_space.sample()

                        next_state, reward, done, _ = self.env.step(action[0])

                        if done:
                            next_state = np.zeros(self.state_dim)
                            reward = -1 if t < 300 else 0
                        else:
                            reward = 0

                        # save experience
                        self.memory.add((state, action[0], reward, next_state))

                        # train using mini batch from experience
                        batch = self.memory.sample_episodes(BATCH)
                        # batch results
                        rewards = list(map(lambda x: x[2], batch))

                        # batch actions
                        actions = list(map(lambda x: x[1], batch))

                        # states = np.array([sample[0] for sample in batch], dtype=np.float32)
                        states = list(map(lambda x: x[0], batch))
                        max_Q, Q = self.brain.predict_Q(sess, states)

                        #no_state = np.zeros(self.state_dim)
                        resultant_states = list(map(lambda x: x[3], batch))
                        max_next_Q, next_Q = self.brain.predict_traget_Q(sess, resultant_states)

                        Q_target = Q
                        # targetQ[0, a[0]] = r + y * maxQ1
                        Q_update = rewards + self.DAMPING_FACTOR * np.max(next_Q, 1)
                        Q_update2 = list(map(lambda x, y: y if y == -1 else x, Q_update, rewards))

                        Q_target2 = list(map(
                            lambda a, q_old, q_new: np.array(
                                [q_new if a == 0 else q_old[0], q_new if a == 1 else q_old[1]]),
                            actions, Q_target, Q_update2))

                        # train and compute gradients
                        _, agent_loss, summary = self.brain.train(sess, x=states, y=Q_target2)
                        writer.add_summary(summary, itr)

                        state = next_state
                        if done:
                            break

                    # Run evaluation after each episode
                    if itr % SPIT_RESULTS == 0:
                        agent_reward, agent_performance, std_rewards, std_performance = self.act(sess, 100)
                        print('Episode: ' + str(int(itr)) + ' Agent performance: ' + str(
                            agent_performance) + ' Loss: ' + str(
                            agent_loss))

                        performance.append(agent_performance)
                        all_mean_reward.append(agent_reward)
                        all_std_rewards.append(std_rewards)
                        all_std_performance.append(std_performance)
                        loss.append(agent_loss)

                        if agent_performance > self.TOPNOTCH:
                            self.TOPMODEL_SAVED = True
                            self.TOPNOTCH = agent_performance
                            save_path = self.brain.saver.save(sess, '../models/problemA7/model.ckpt')

                    # update the target network
                    if itr % 5 == 0 and itr is not 0:
                        self.brain.update_target_network(sess)
                        print('traget network updated..')

                plt.figure(1, figsize=(14, 10), dpi=50)
                plt.subplot(311)
                plotGraph(data=performance[0:-1],
                          title='Agent Performance (avg 100) - (learning rate ' + str(LEARNING_RATE) + ')',
                          color='b', ylabel='Episode Length', label='Episode Length')
                plt.subplot(312)
                plotGraph(data=all_mean_reward[0:-1],
                          title='Agent Rewards (avg 100) - (learning rate ' + str(LEARNING_RATE) + ')',
                          color='g', ylabel='Rewards', label='rewards')
                plt.subplot(313)
                plotGraph(data=loss[0:-1], title='Agent Training Loss - (learning rate ' + str(LEARNING_RATE) + ')',
                          color='r', ylabel='Bellman Loss', label='loss')
                plt.xlabel('Episodes', fontsize=18)
                if SAVE_PLOT:
                    plt.savefig('../plots/problemA7/plot-' + str(LEARNING_RATE) + '.eps', format='eps', dpi=50)
                else:
                    plt.show()

                if not self.TOPMODEL_SAVED:
                    save_path = self.brain.saver.save(sess, '../models/problemA7/model.ckpt')
                    print('Model weights saved in file: ', save_path)
            else:
                # test mode
                print('\n\n---Running in test mode---')
                self.brain.saver.restore(sess, '../models/problemA7/model.ckpt')
                agent_reward, agent_performance, _, _ = self.act(sess, 100)
                print('Average Agent Reward: ' + str(
                    agent_reward) + ', Average Agent performance (episode length): ' + str(agent_performance))
                performance_dict = {'Avg Score (reward)': [agent_reward],
                                    'Avg episode Counts': [agent_performance]}
                pd.DataFrame(performance_dict).to_csv('../outputs/problemA7-' + str(LEARNING_RATE) + '.csv')


def plotGraph(data, label, color, ylabel, title, sd=None):
    x = np.arange(0, 2000, 20)
    plt.plot(x, data, color=color, label=label, lw=2)
    if sd is not None:
        plt.fill_between(x=x, y1=np.add(data, sd), y2=np.subtract(data, sd), alpha=0.1)
    plt.grid(True)
    #plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    #plt.title(title, fontsize=22, fontweight='bold')
    plt.legend()


def main(agent):
    np.random.seed(SEED)
    random.seed(a=SEED)

    env = gym.make(agent)
    for lr in LEARNING_RATES:
        print('#Agent = Learning rate: ' + str(lr) + ' #Network= ' + 'Linear:' + str(LINEAR))
        agent = Agent(env, LINEAR, lr)
        agent.replay(lr)


if __name__ == '__main__':
    args = sys.argv
    if len(args) > 1:
        TRAIN_MODE = args[1].lower() == 'train'
        if len(args) > 2:
            BATCH = int(args[2])
        if len(args) > 3:
            LEARNING_RATES = [float(args[3])]

    else:
        TRAIN_MODE = True  # todo update to False
    main(AGENT_CARTPOLE_300)

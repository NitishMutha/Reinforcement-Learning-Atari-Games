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
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.exposure import adjust_gamma

AGENT_CARTPOLE = 'CartPole-v0'
AGENT_CARTPOLE_300 = 'CartPole-300-v0'
AGENT_PACMAN = 'MsPacman-v3'
AGENT_PONG = 'Pong-v3'
AGENT_BOXING = 'Boxing-v3'

OBSERVATION_DIM = 2304
IMAGE_DIM = 48

LINEAR = False
RMS_OPTIMIZER = True

RENDER_AGENT = False
N_EPISODES = 1000000 #1200000  # 1 milliom
EXPERIENCE_BUFFER = 100000#100000 #200000
SPIT_RESULTS = 50000 #50000  # 50k
TARGET_NETWORK_UPDATE_RATE = 5000 #5000
SEED = 0
FRAME_STACK = 4
DAMPING_FACTOR = 0.99

BATCH = 32
LEARNING_RATES = [0.0001]  # [0.001, 0.0001, 0.00001, 0.01, 0.1, 0.5]
HIDDEN_LAYER = 100

RELOAD_BUFFER = False
RELOAD_MODEL = True
TRAIN_MODE = True
SAVE_PLOT = True
MODE = 1
LOG_PATH = '../summary/problemBpong'


# writer = tf.summary.FileWriter(LOG_PATH)


class Brain:
    def __init__(self, state_dim, n_actions, LEARNING_RATE):
        tf.reset_default_graph()
        tf.set_random_seed(SEED)
        self.target_update_rate = 1

        # create computation graph
        self.state_vectors = tf.cast(tf.placeholder(shape=[None, state_dim * FRAME_STACK], dtype=tf.uint8,
                                                    name='state_vectors'), tf.float32)
        self.next_state_vectors = tf.cast(tf.placeholder(shape=[None, state_dim * FRAME_STACK], dtype=tf.uint8,
                                                         name='next_state_vectors'), tf.float32)

        with tf.variable_scope('q_network'):
            states = tf.transpose(tf.reshape(self.state_vectors, [-1, FRAME_STACK, IMAGE_DIM, IMAGE_DIM]), [0, 2, 3, 1])
            self.Q = self.get_q_network(states, n_actions)

        with tf.variable_scope('target_network'):
            states = tf.transpose(tf.reshape(self.next_state_vectors, [-1, FRAME_STACK, IMAGE_DIM, IMAGE_DIM]),
                                  [0, 2, 3, 1])
            self.target_network = tf.stop_gradient(
                self.get_q_network(states, n_actions))

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

        # tf.summary.scalar("loss", self.loss)
        # self.merged_summary = tf.summary.merge_all()

        self.tf_init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

    def get_q_network(self, state, n_actions):

        input_layer = tf.reshape(state, [-1, IMAGE_DIM, IMAGE_DIM, 4])

        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            strides=(4, 4),
            kernel_size=8,
            padding="valid",
            activation=tf.nn.relu)

        conv2 = tf.layers.conv2d(
            inputs=conv1,
            filters=64,
            strides=(2, 2),
            kernel_size=4,
            padding="valid",
            activation=tf.nn.relu)

        conv3 = tf.layers.conv2d(
            inputs=conv1,
            filters=64,
            strides=(1, 1),
            kernel_size=3,
            padding="valid",
            activation=tf.nn.relu)

        conv2flat = tf.reshape(conv3, [-1, 9 * 9 * 64])
        dense = tf.layers.dense(inputs=conv2flat, units=256, activation=tf.nn.relu)

        Q = tf.layers.dense(inputs=dense, units=n_actions)

        return Q

    def predict_Q(self, sess, x):
        return sess.run([self.predict, self.Q], feed_dict={self.state_vectors: x})

    def predict_traget_Q(self, sess, x):
        return sess.run([self.predict_target_network, self.target_network], feed_dict={self.next_state_vectors: x})

    def train(self, sess, x, y):
        return sess.run([self.optimize, self.loss],
                        feed_dict={self.state_vectors: x, self.next_Q: y})

    def update_target_network(self, sess):
        return sess.run([self.target_network_update])


class Memory:
    def __init__(self, limit):
        self.experienceBuffer = deque(maxlen=limit)
        self.limit = limit
        self.episode_lens = np.array([])
        self.frame_buffer = deque(maxlen=FRAME_STACK)
        self.frame_buffer_test = deque(maxlen=FRAME_STACK)

    def add(self, sample):
        self.experienceBuffer.append(sample)

    def sample_episodes(self, batch):
        batch = min(batch, len(self.experienceBuffer))
        return random.sample(tuple(self.experienceBuffer), batch)

    def add_frame(self, f, times=1):
        for tt in range(times):
            self.frame_buffer.append(f.astype(np.uint8))

    def add_frame_test(self, f, times=1):
        for tt in range(times):
            self.frame_buffer_test.append(f.astype(np.uint8))

    def stack_frames(self):
        return np.array(list(self.frame_buffer))  # .astype('uint8')

    def stack_test_frames(self):
        return np.array(list(self.frame_buffer_test))  # .astype('uint8')


class Agent:
    DAMPING_FACTOR = 0.99
    EPSILON = np.float32(1)

    def __init__(self, env, LEARNING_RATE):
        self.env = env
        self.state_dim = OBSERVATION_DIM
        self.n_action = self.env.action_space.n
        self.brain = Brain(self.state_dim, self.n_action, LEARNING_RATE)
        self.memory = Memory(EXPERIENCE_BUFFER)
        self.steps = 0
        self.MAX_EPISODE_LEN = env._max_episode_steps

    def rewardClip(self, r):
        if r >= 1:
            return 1.
        elif r <= -1:
            return -1.
        else:
            return 0.

    def preprocess(self, I):
        # 210x160x3 uint8 frame into 6400 (80x80) 1D float vector
        # plt.figure()
        # plt.imshow(I)
        # plt.show()
        if AGENT == AGENT_PONG:
            I = I[35:195]  # crop
            I[I == 144] = 0  # erase background (background type 1)
            I[I == 109] = 0  # erase background (background type 2)
            I = adjust_gamma(I, 2, 1)
            I = resize(I, (120, 120), mode='reflect')
            I = resize(I, (90, 90), mode='reflect')
            # I = resize(I, (60, 60), mode='reflect')
            # I = resize(I, (40, 40), mode='reflect')
            I = resize(I, (IMAGE_DIM, IMAGE_DIM), mode='reflect', preserve_range=True)
            I = rgb2gray(I)
            return I.astype(np.float32).ravel()
        elif AGENT == AGENT_PACMAN:
            I = I[4:168]  # crop
            I[I == 28] = 0  # erase background (background type 1)
            I[I == 136] = 0  # erase background (background type 2)
            I = resize(I, (160, 160), mode='reflect')
            I = resize(I, (128, 128), mode='reflect')
            # I = resize(I, (64, 64), mode='reflect')
            I = resize(I, (IMAGE_DIM, IMAGE_DIM), mode='reflect', preserve_range=True)
            I = rgb2gray(I)
            return I.astype(np.float32).ravel()
        elif AGENT == AGENT_BOXING:
            I = I[35:178, 10:153]  # crop
            I[I == 156] = 0  # erase background (background type 1)
            I[I == 66] = 0  # erase background (background type 2)
            I = adjust_gamma(I, 1, 2)
            I = resize(I, (IMAGE_DIM, IMAGE_DIM), mode='reflect', preserve_range=True)
            I = rgb2gray(I)
            return I.astype(np.float32).ravel()

    # perform action
    def act(self, sess, iterations):

        allRewards = []
        allExperitmentLengths = []
        allScore = []

        for itz in range(iterations):
            obs_test = self.preprocess(self.env.reset())
            # think to send just one frame
            self.memory.add_frame_test(obs_test, 4)
            state = self.memory.stack_test_frames()

            episode_reward = 0
            score = 0

            for t in range(self.MAX_EPISODE_LEN):
                if RENDER_AGENT: self.env.render()

                action, _ = self.brain.predict_Q(sess, np.reshape(state, (1, -1)))

                next_obs_test, reward, done, _ = self.env.step(action[0])
                next_obs_test_processed = self.preprocess(next_obs_test)
                self.memory.add_frame_test(next_obs_test_processed)

                state = self.memory.stack_test_frames()

                reward = self.rewardClip(reward)
                episode_reward += reward * self.DAMPING_FACTOR ** t
                score += reward

                if done:
                    allRewards.append(episode_reward)
                    allExperitmentLengths.append([t + 1])
                    allScore.append(score)
                    break
        return np.mean(allRewards), np.mean(allExperitmentLengths), np.std(allRewards), np.std(
            allExperitmentLengths), np.mean(allScore), np.std(allScore)

    # save state
    def createBuffer(self, sess, maxIterations):

        for itz in range(maxIterations):
            obs = self.preprocess(self.env.reset())
            self.memory.add_frame(obs, 4)
            state = self.memory.stack_frames()

            for t in range(self.MAX_EPISODE_LEN):
                if RENDER_AGENT:
                    self.env.render()

                # action, _ = self.brain.predict_Q(sess, np.reshape(state, (1, -1)))
                action = [self.env.action_space.sample()]
                next_obs, reward, done, _ = self.env.step(action[0])
                next_obs_processed = self.preprocess(next_obs)

                # if done:
                #     next_obs_processed = np.zeros((self.state_dim))

                self.memory.add_frame(next_obs_processed)

                next_state = self.memory.stack_frames()

                reward = self.rewardClip(reward)

                self.memory.add((state, action[0], reward, next_state))

                # state = next_state

                if len(self.memory.experienceBuffer) == EXPERIENCE_BUFFER:
                    return
                if done:
                    break

    def get_next_target(self, actions_, Qtarget_, Qupdate_):
        if AGENT == AGENT_PONG:
            return list(map(
                lambda a, q_old, q_new: np.array(
                    [q_new if a == 0 else q_old[0], q_new if a == 1 else q_old[1],
                     q_new if a == 2 else q_old[2], q_new if a == 3 else q_old[3],
                     q_new if a == 4 else q_old[4], q_new if a == 5 else q_old[5]]),
                actions_, Qtarget_, Qupdate_))
        elif AGENT == AGENT_PACMAN:
            return list(map(
                lambda a, q_old, q_new: np.array(
                    [q_new if a == 0 else q_old[0], q_new if a == 1 else q_old[1],
                     q_new if a == 2 else q_old[2], q_new if a == 3 else q_old[3],
                     q_new if a == 4 else q_old[4], q_new if a == 5 else q_old[5],
                     q_new if a == 6 else q_old[6], q_new if a == 7 else q_old[7],
                     q_new if a == 8 else q_old[8]]),
                actions_, Qtarget_, Qupdate_))
        else:  # boxing
            return list(map(
                lambda a, q_old, q_new: np.array(
                    [q_new if a == 0 else q_old[0], q_new if a == 1 else q_old[1],
                     q_new if a == 2 else q_old[2], q_new if a == 3 else q_old[3],
                     q_new if a == 4 else q_old[4], q_new if a == 5 else q_old[5],
                     q_new if a == 6 else q_old[6], q_new if a == 7 else q_old[7],
                     q_new if a == 8 else q_old[8], q_new if a == 9 else q_old[9],
                     q_new if a == 10 else q_old[10], q_new if a == 11 else q_old[11],
                     q_new if a == 12 else q_old[12], q_new if a == 13 else q_old[13],
                     q_new if a == 14 else q_old[14], q_new if a == 15 else q_old[15],
                     q_new if a == 16 else q_old[16], q_new if a == 17 else q_old[17]
                     ]),
                actions_, Qtarget_, Qupdate_))

    # train
    def replay(self, LEARNING_RATE):

        with tf.Session() as sess:
            sess.run(self.brain.tf_init)
            # writer.add_graph(sess.graph)

            frame_counts = []
            std_frame_counts = []

            final_rewards = []
            loss = []
            final_score = []

            STEPS = 0
            lastScore = -50
            LOOP_BREAK = False
            if TRAIN_MODE:
                print('\n\n---Running in train mode---')
                global_steps = 500
                decay_steps = 500

                # create a buffer

                print('Creating experience buffer...')
                self.createBuffer(sess, EXPERIENCE_BUFFER)
                print('Buffer creation complete...')

                print('Training...')
                for itr in range(N_EPISODES):

                    obs = self.preprocess(self.env.reset())
                    self.memory.add_frame(obs, 4)
                    state = self.memory.stack_frames()
                    action, Q = self.brain.predict_Q(sess, np.reshape(state, (1, -1)))

                    for t in range(self.MAX_EPISODE_LEN):

                        if RENDER_AGENT: self.env.render()

                        # Exploration by e-greedy
                        if np.random.rand(1) < self.EPSILON:
                            action[0] = self.env.action_space.sample()

                        if self.EPSILON > 0.1:
                            global_steps += 1
                            self.EPSILON = self.DAMPING_FACTOR ** (global_steps / decay_steps)
                        else:
                            self.EPSILON = 0.1

                        next_obs, reward, done, _ = self.env.step(action[0])
                        next_state_processed = self.preprocess(next_obs)

                        # if done:
                        #     next_state_processed = np.zeros((self.state_dim))

                        self.memory.add_frame(next_state_processed)
                        next_state = self.memory.stack_frames()

                        reward = self.rewardClip(reward)

                        # save experience
                        self.memory.add((state, action[0], reward, next_state))

                        # train using mini batch from experience
                        batch = self.memory.sample_episodes(BATCH)
                        batchLen = len((batch))

                        # batch results
                        rewards = list(map(lambda x: x[2], batch))

                        # batch actions
                        actions = list(map(lambda x: x[1], batch))

                        # states
                        states = list(map(lambda x: x[0], batch))

                        # next states
                        resultant_states = list(map(lambda x: x[3], batch))

                        straight_states = np.reshape(states, (BATCH, -1))
                        max_Q, Q = self.brain.predict_Q(sess, straight_states)

                        resultant_straight_states = np.reshape(resultant_states, (BATCH, -1))
                        max_next_Q, next_Q = self.brain.predict_traget_Q(sess, resultant_straight_states)

                        if np.random.rand(1) < self.EPSILON:
                            max_next_Q[0] = self.env.action_space.sample()

                        # experimenting
                        Q_target = np.copy(Q)

                        Q_update = rewards + self.DAMPING_FACTOR * np.max(next_Q, 1)
                        # Q_update2 = list(map(lambda x, y: y if y == -1 else x, Q_update, rewards))

                        Q_target2 = self.get_next_target(actions, Q_target, Q_update)

                        # train and compute gradients
                        _, agent_loss = self.brain.train(sess, x=np.reshape(states, (BATCH, -1)),
                                                         y=Q_target2)
                        # writer.add_summary(summary, itr)

                        loss.append(agent_loss)

                        state = next_state
                        action = max_next_Q

                        # Run evaluation after each 50k steps
                        if STEPS % SPIT_RESULTS == 0:
                            print('Evalating on agent..')
                            agent_reward, agent_performance, std_reward, std_performance, mean_episode_score, std_episode_score = self.act(
                                sess, 2)
                            print(str(STEPS / SPIT_RESULTS) + '  Agent Frames: ' + str(agent_performance) +
                                  ' Agent Reward: ' + str(agent_reward) +
                                  ' Agent Score: ' + str(mean_episode_score) +
                                  ' Agent loss: ' + str(agent_loss))

                            frame_counts.append(agent_performance)
                            std_frame_counts.append(std_performance)
                            final_rewards.append(agent_reward)
                            final_score.append(mean_episode_score)

                            if mean_episode_score > lastScore: self.brain.saver.save(sess,
                                                                                     '../models/' + AGENT + '/model.ckpt')

                        # update the target network
                        if STEPS % TARGET_NETWORK_UPDATE_RATE == 0 and itr is not 0:
                            self.brain.update_target_network(sess)
                            print('target network updated..')

                        STEPS += 1

                        if STEPS == 1000000:
                            LOOP_BREAK = True
                            break

                        if done: break

                    if LOOP_BREAK: break

                plotGraph(data=frame_counts,
                          title='Agent Frame counts - (learning rate ' + str(LEARNING_RATE) + ')',
                          xlabel='Steps', ylabel='Frame counts', label='frame counts', sd=std_frame_counts)
                plotGraph(data=final_rewards,
                          title='Agent Avg Cumulative Reward - (learning rate ' + str(LEARNING_RATE) + ')',
                          xlabel='Steps', ylabel='Avg cumulative reward', label='avg cumulative reward')

                plotGraph(data=loss,
                          title='Agent Training Loss - (learning rate ' + str(LEARNING_RATE) + ')',
                          xlabel='Steps', ylabel='Loss', label='training loss')

                plotGraph(data=final_score,
                          title='Agent Training Score - (learning rate ' + str(LEARNING_RATE) + ')',
                          xlabel='Steps', ylabel='Score', label='score')

                save_path = self.brain.saver.save(sess, '../models/' + AGENT + '/model.ckpt')
                print('Model weights saved in file: ', save_path)

                np.savez('../models/problemB/' + AGENT + '.npz', final_rewards=final_rewards,
                         frame_counts=frame_counts, loss=loss, final_score=final_score)

            else:
                # test mode
                print('\n\n---Running in test mode---')
                self.brain.saver.restore(sess, '../models/' + AGENT + '/model.ckpt')
                agent_reward, agent_performance, std_reward, std_performance, mean_episode_score, std_episode_score = self.act(
                    sess, 100)
                print('Average Agent Reward: ' + str(
                    agent_reward) + ', Average Agent performance (frame count): ' + str(
                    agent_performance) + ', Average score: ' + str(mean_episode_score))

                performance_dict = {'Avg Score (Cumulative discounted reward)': [agent_reward],
                                    'Avg Frame Counts': [agent_performance],
                                    'Std Score (Cumulative discounted reward)': [std_reward],
                                    'Std Frame Counts': [std_performance],
                                    'Avg Score': [mean_episode_score],
                                    'Std Score': [std_episode_score]
                                    }
                pd.DataFrame(performance_dict).to_csv('../outputs/B_3_' + AGENT + '-1.csv')


def plotGraph(data, label, xlabel, ylabel, title, sd=None):
    plt.figure(figsize=(14, 10), dpi=50)
    plt.plot(data, 'b', label=label, lw=2)
    if sd is not None:
        x = np.arange(0, len(data), 1)
        plt.fill_between(x=x, y1=np.add(data, sd), y2=np.subtract(data, sd), alpha=0.4)
    plt.grid(True)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.title(title, fontsize=22, fontweight='bold')
    plt.legend()
    if SAVE_PLOT:
        plt.savefig('../plots/' + AGENT + '/plot-' + title + '.eps', format='eps', dpi=50)
    else:
        plt.show()


def get_random_performance(env):
    agent = Agent(env, 0.001)

    all_rewards = []
    episode_frame_count = []
    all_score = []
    for i_episode in range(100):

        observation = env.reset()
        episode_reward = 0
        score = 0

        for t in range(env._max_episode_steps):
            if RENDER_AGENT: env.render()

            action = env.action_space.sample()
            observation, reward, done, _ = env.step(action)

            reward = agent.rewardClip(reward)

            episode_reward += reward * (DAMPING_FACTOR ** t)
            score += reward
            if done:
                episode_frame_count.append(t + 1)
                all_rewards.append(episode_reward)
                all_score.append(score)
                break
    print('Avg Score (Cumulative discounted reward): ' + str(np.mean(all_rewards)) + ' Avg Frame Counts: ' + str(
        np.mean(episode_frame_count)))
    print('SD Score (Cumulative discounted reward): ' + str(np.std(all_rewards)) + ' SD Frame Counts: ' + str(
        np.std(episode_frame_count)))
    print('Avg Score (Raw): ' + str(np.mean(all_score)) + ' SD Score: ' + str(
        np.std(all_score)))
    performance_dict = {'Avg Score (Cumulative discounted reward)': [np.mean(all_rewards)],
                        'Avg Frame Counts': [np.mean(episode_frame_count)],
                        'SD Score (Cumulative discounted reward)': [np.std(all_rewards)],
                        'SD Frame Counts': [np.std(episode_frame_count)],
                        'Avg Score (raw)': [np.mean(all_score)],
                        'SD Score (raw)': [np.std(all_score)]
                        }
    pd.DataFrame(performance_dict).to_csv('../outputs/B_' + AGENT + '-1.csv')


def get_untrained_performance(env):
    agent = Agent(env, 0.001)

    with tf.Session() as sess:
        sess.run(agent.brain.tf_init)

        avg_reward, avg_frame, reward_std, frame_std, mean_episode_score, std_episode_score = agent.act(sess, 100)

        print(' Agv Agent Reward: ' + str(avg_reward) +
              ' Avg Agent Frames: ' + str(avg_frame) +
              ' SD Agent Reward: ' + str(reward_std) +
              ' SD Agent Frames: ' + str(frame_std) +
              ' Agv Agent Score: ' + str(mean_episode_score) +
              ' SD Agent Score: ' + str(std_episode_score)
              )
        performance_dict = {'Agv Agent Reward': [avg_reward],
                            'Avg Agent Frames': [avg_frame],
                            'SD Agent Reward': [reward_std],
                            'SD Agent Frames': [frame_std],
                            'Agv Agent Score': [mean_episode_score],
                            'SD Agent Score': [std_episode_score],
                            }
        pd.DataFrame(performance_dict).to_csv('../outputs/B_' + AGENT + '-2.csv')


def main(agent):
    np.random.seed(SEED)
    random.seed(a=SEED)

    env = gym.make(agent)

    if MODE == 1:
        get_random_performance(env)

    elif MODE == 2:
        get_untrained_performance(env)

    elif MODE > 2:

        lr = LEARNING_RATES[0]
        print('#Agent = Learning rate: ' + str(lr))
        agent = Agent(env, lr)
        agent.replay(lr)


if __name__ == '__main__':
    selectedAgent = ''

    args = sys.argv
    if len(args) > 1:
        selectedAgent = args[1]  # agent name boxing, pacman, pong
        if len(args) > 2:
            MODE = int(args[2])  # Question mode
        if len(args) > 3:
            TRAIN_MODE = args[3].lower() == 'train'  # train/test mode
        if len(args) > 4:
            BATCH = int(args[4])  # batch
        if len(args) > 5:
            LEARNING_RATES = [float(args[5])]  # learning rate
    else:
        TRAIN_MODE = True  # todo update to False
        MODE = 3

    if selectedAgent == 'boxing':
        AGENT = AGENT_BOXING
    elif selectedAgent == 'pacman':
        AGENT = AGENT_PACMAN
    elif selectedAgent == 'pong':
        AGENT = AGENT_PONG
    else:
        AGENT = AGENT_BOXING
    main(AGENT)

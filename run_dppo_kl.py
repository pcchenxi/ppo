import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import threading, queue
import math
import pickle, os
import gym

from kl.utils import Scaler
from environment import centauro_env

from dppo.policy import Policy
from dppo.value_function import NNValueFunction
from dppo import ppo_functions as pf

MODE = 'training'
# MODE = 'testing'

EP_MAX = 900000000
EP_LEN = 40
N_WORKER = 4                 # parallel workers
GAMMA = 0.98                 # reward discount factor
LAM = 0.0

BATCH_SIZE = 1000

###############################

S_DIM, A_DIM = centauro_env.observation_space, centauro_env.action_space         # state and action dimension
SCALER = Scaler(S_DIM)

ep_dir = './batch/'
# N_image_size = centauro_env.observation_image_size
# N_robot_state_size = centauro_env.observation_control

class PPO(object):
    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        self.summary_writer = tf.summary.FileWriter('data/log', self.sess.graph)

        kl_targ = 0.009
        self.val_func = NNValueFunction(S_DIM, self.sess, self.summary_writer)
        self.policy = Policy(S_DIM, A_DIM, kl_targ, 'clip', self.sess, self.summary_writer)

        #############################################################################################
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        # self.load_model()

    def load_model(self):
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state('./model/rl/')
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print ('loaded')
        else:
            print ('no model file')  

    def update(self):
        global GLOBAL_UPDATE_COUNTER, GLOBAL_EP, SCALER, INIT_DONE
        while not COORD.should_stop():
            if GLOBAL_EP < EP_MAX:
                UPDATE_EVENT.wait()                     # wait until get batch of data
                print('')
                print('update!')

                data = [QUEUE.get() for _ in range(QUEUE.qsize())]      # collect data from all workers                
                data = np.vstack(data)[:, 0]
                trajectories = []
                for t in data:
                    rewards = t['rewards']
                    if len(rewards) > 1:
                        trajectories.append(t)

                unscaled = np.concatenate([t['unscaled_obs'] for t in trajectories])
                SCALER.update(unscaled) 

                if INIT_DONE == False:
                    pf.add_value(trajectories, self.val_func)  # add estimated values to episodes
                    pf.add_disc_sum_rew(trajectories, GAMMA)  # calculated discounted sum of Rs
                    pf.add_gae(trajectories, GAMMA, LAM)  # calculate advantage
                    # concatenate all episodes into single NumPy arrays
                    observes, actions, advantages, disc_sum_rew = pf.build_train_set(trajectories)
                    # add various stats to training log:

                    # update
                    self.policy.update(observes, actions, advantages, GLOBAL_EP)  # update policy
                    self.val_func.fit(observes, disc_sum_rew, GLOBAL_EP)  # update value function
                else:
                    INIT_DONE = True

                UPDATE_EVENT.clear()        # updating finished
                GLOBAL_UPDATE_COUNTER = 0   # reset counter
                ROLLING_EVENT.set()         # set roll-out available

                self.saver.save(self.sess, './model/rl/model.cptk') 
    

    def choose_action(self, s):
        a = self.policy.sample(s).reshape((1, -1)).astype(np.float32)   
        return np.clip(a, -1, 1)

    def get_v(self, s):
        value = self.val_func.predict(s)
        return value 

class Worker(object):
    def __init__(self, wid):
        self.wid = wid
        self.env = centauro_env.Simu_env(20000 + wid)
        self.ppo = GLOBAL_PPO


    def write_summary(self, summary_name, value):
        global GLOBAL_EP
        # if INIT_DONE == False:
        #     return
        summary = tf.Summary()
        summary.value.add(tag=summary_name, simple_value=float(value))
        self.ppo.summary_writer.add_summary(summary, GLOBAL_EP)
        self.ppo.summary_writer.flush()   

    def varifly_values(self):
        global MODE, GLOBAL_EP
        grid_size = 0.5        
        img_size = int(2/grid_size)
        map_shift = 1
        img = np.zeros((img_size, img_size), np.float32)
        values = []
        for i in np.arange(-1, 1, grid_size):
            for j in np.arange(-1, 1, grid_size):
                self.env.call_sim_function('centauro', 'move_robot', [i, j])
                s, r, done, info = self.env.step([0,0,0,0,0])
                value = self.ppo.get_v(s)[0, 0]

                # value = np.clip(value, -1, 2)
                # if value > 1:
                #     print ('state', s[0], s[1], value)
                x = i + map_shift
                y = j + map_shift
                row = img_size - int(y/grid_size) -1
                col = int(x/grid_size)
                # print(i, j, row, col)
                img[row, col] = value #(value+1)/2 * 255
                values.append(value)
                
        summary = tf.Summary()
        summary.value.add(tag='Perf/Average Net_Value', simple_value=float(np.mean(values)))
        self.ppo.summary_writer.add_summary(summary, GLOBAL_EP)
        self.ppo.summary_writer.flush()   

        print(img)
        if MODE == 'testing':
            img = (img + 2.5)/5 * 255
            plt.clf()
            plt.imshow(img, cmap='gray')
            # plt.imshow(decoded_grid[0,:,:,0], cmap='gray')
            # plt.pause(0.001)
            plt.show()

    def work(self):
        global GLOBAL_EP, GLOBAL_UPDATE_COUNTER, BATCH_SIZE, SUCCESS_NUM, CRASH_NUM, SCALER
        while not COORD.should_stop():
            obs = self.env.reset()
            observes, actions, rewards, unscaled_obs = [], [], [], []
            stuck_num = 0
            step = 0.0
            scale, offset = SCALER.get()
            scale[-1] = 1.0  # don't scale time step feature
            offset[-1] = 0.0  # don't offset time step feature
            for t in range(EP_LEN):
                if not ROLLING_EVENT.is_set():                  # while global PPO is updating
                    ROLLING_EVENT.wait()                        # wait until PPO is updated
                    GLOBAL_UPDATE_COUNTER -= len(rewards)
                    observes, actions, rewards = [], [], []   # clear history buffer, use new policy to collect data
                    # break

                u_obs = obs
                obs = obs.astype(np.float32).reshape((1, -1))
                # obs = np.append(obs, [[step]], axis=1)  # add time step feature
                unscaled_obs.append(obs)
                obs = (obs - offset) * scale  # center and scale observations
                observes.append(obs)
                action = self.ppo.choose_action(obs)
                actions.append(action)
                # print(action)
                obs_, reward, done, _ = self.env.step(np.squeeze(action, axis=0))

                # check if the robot played too much with the obstacle
                # stuck_limit = 5
                # max_diff = np.amax(np.abs(u_obs[-5:]-obs_[-5:]))                
                # if max_diff == 0:
                #     # print(max_diff)
                #     # print(u_obs[-5:])
                #     # print(obs_[-5:])       
                #     reward -= 0.1         
                #     stuck_num += 1
                # else:
                #     stuck_num == 0

                # if stuck_num > stuck_limit:
                #     reward -= -0.1
                #     done = -1

                rewards.append(reward)
                obs = obs_
                step += 1e-3  # increment time step feature

                GLOBAL_UPDATE_COUNTER += 1               # count to minimum batch size, no need to wait other workers
   
                # print('GLOBAL_UPDATE_COUNTER', GLOBAL_UPDATE_COUNTER/BATCH_SIZE, end="\r")
                if t == EP_LEN - 1 or GLOBAL_UPDATE_COUNTER >= BATCH_SIZE or done!= 0:
                    if done == 1:
                        SUCCESS_NUM += 1
                        print('goal')
                        self.write_summary('Perf/ep_length', t)  
                    elif done == -1:
                        CRASH_NUM += 1        
                        print('crash')
                        self.write_summary('Perf/ep_length', t)  
                    elif done == 0:
                        obs_ = obs_.astype(np.float32).reshape((1, -1))
                        # obs_ = np.append(obs_, [[step]], axis=1)  # add time step feature
                        unscaled_obs.append(obs_)
                        obs_ = (obs_ - offset) * scale  # center and scale observations
                        rewards[-1] += GAMMA*self.ppo.get_v(obs_) 

                        print('unfinish')
                        
                    mean_reward = np.sum(rewards[:-1])
                    self.write_summary('Perf/mean_reward', mean_reward)  
                    
                    bs, ba, br, bus = np.concatenate(observes), np.concatenate(actions), np.array(rewards, dtype=np.float64), np.concatenate(unscaled_obs)

                    trajectory = {'observes': bs,
                                'actions': ba,
                                'rewards': br,
                                'unscaled_obs': bus}

                    observes, actions, rewards, unscaled_obs = [], [], [], []
                    QUEUE.put(trajectory)          # put data in the queue
                    if GLOBAL_UPDATE_COUNTER >= BATCH_SIZE:
                        ROLLING_EVENT.clear()       # stop collecting data
                        UPDATE_EVENT.set()          # globalPPO update

                    if GLOBAL_EP >= EP_MAX:         # stop training
                        COORD.request_stop()
                    
                    GLOBAL_EP += 1          
                    ep_length = 51
                    if GLOBAL_EP%ep_length == 0:
                        self.write_summary('Perf/Success Rate', SUCCESS_NUM/ep_length)  
                        self.write_summary('Perf/Crash Rate', CRASH_NUM/ep_length)  
                        print('')      
                        print('success rate', SUCCESS_NUM, SUCCESS_NUM/ep_length)          
                        SUCCESS_NUM = 0       
                        CRASH_NUM = 0
                        # if self.wid == 0:
                        #     self.varifly_values()
                                                        
                    if done!= 0:
                        # GLOBAL_EP += 1
                        break

if __name__ == '__main__':
    GLOBAL_PPO = PPO()
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    UPDATE_EVENT.clear()            # not update now
    ROLLING_EVENT.set()             # start to roll out

    GLOBAL_UPDATE_COUNTER, GLOBAL_EP, SUCCESS_NUM, CRASH_NUM = 1, 1, 0, 0
    INIT_DONE = False

    if MODE == 'training':
        workers = []
        for i in range(N_WORKER):
            workers.append(Worker(i))
        
        COORD = tf.train.Coordinator()
        QUEUE = queue.Queue()           # workers putting data in this queue
        threads = []
        
        for worker in workers:          # worker threads
            t = threading.Thread(target=worker.work, args=())
            t.start()                   # training
            threads.append(t)
        # add a PPO updating thread
        threads.append(threading.Thread(target=GLOBAL_PPO.update,))
        threads[-1].start()
        COORD.join(threads)

    # checking value function
    if MODE == 'testing':
        env = centauro_env.Simu_env(20000)
        worker = Worker(env, 0)
        worker.ppo.load_model()
        worker.varifly_values()

from functions.LSTM_functions import ReplayBufferLSTM2
from functions.GRU_functions import ReplayBufferGRU
from RL.env import *


env = AdaptForLSTM()
state_space = env.observation_space
action_space = env.action_space
action_dim = action_space.shape[0]
action_range = 1.



replay_buffer_size = 100
replay_buffer = ReplayBufferLSTM2(replay_buffer_size)
#replay_buffer = ReplayBufferGRU(replay_buffer_size)
hidden_dim= 128
gamma= 0.995
soft_q_lr = 0.0015
policy_lr = 0.003  
alpha_lr = 0.002 
batch_size = 10
update_itr= 1
reward_scale= 10
target_entropy= -2
soft_tau = 0.015
train_freq = 100
batch_size = 10


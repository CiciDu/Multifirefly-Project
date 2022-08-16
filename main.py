from process_monkey_data import*
from useful_funcs import*


# for monkey
NEW_DATASET = False
MONKEY_DATA = True
NO_PLOT_NEEDED = True
data_folder_name = "0219"
data_num = 19

# for agent
NEW_DATASET = True
MONKEY_DATA = False
NO_PLOT_NEEDED = True
data_folder_name = "LSTM_July_29"
data_num = 721
trial_total_num = 30

import numpy as np
np.random.seed(7777)
#rng = np.random.default_rng(2021) 
NEW_DATASET = True
MONKEY_DATA = True
SHOW_INTERESTING = False
trial_total_num = 20
list_of_colors = ["navy", "magenta", "white", "gray", "brown", "black"] # For plotting ff clusters
point_index_array = np.arange(1,100,10)

"""### Connect to Google Drive"""

! pip install google.colab
from google.colab import drive
drive.mount('/content/gdrive')

"""### unzip data"""

!pip install neo
import neo
import os.path
from os import path

#!unzip gdrive/MyDrive/behavioural_data.zip
!unzip gdrive/MyDrive/fireflies_data/0219.zip
#!unzip gdrive/MyDrive/fireflies_data/0220.zip
#!unzip gdrive/MyDrive/fireflies_data/0221.zip
data_folder_name = "0219"
data_num = 19
#seg_reader = neo.io.Spike2IO(filename="/content/behavioural_data/m51s26.smr").read_segment()
seg_reader = neo.io.Spike2IO(filename="/content/0219/m51c0936.smr").read_segment()

"""### Import packages"""

import os.path
from os import path

import numpy as np
import matplotlib


matplotlib.rcParams.update(matplotlib.rcParamsDefault)

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from IPython.display import HTML
import pandas as pd
import math
import collections
import re
import os, sys
from contextlib import contextmanager
from scipy.signal import decimate
from matplotlib import rc, cm
from sklearn.linear_model import LinearRegression
import torch
from math import pi
torch.set_printoptions(sci_mode=False)
np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
import seaborn as sns

!pip install matplotlib-scalebar
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.ticker as ticker

from numpy import linalg as LA
import csv 
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering
import plotly.express as px
from random import randint




"""### Animation"""

rc('animation', html='jshtml')
matplotlib.rcParams['animation.embed_limit'] = 2**128

"""### smr_extractor"""


Channel_signal_output,marker_list,smr_sampling_rate = smr_extractor().extract_data()
#Considering the first smr file, use marker_list[0], Channel_signal_output[0]
juice_timestamp = marker_list[0]['values'][marker_list[0]['labels']==4]
Channel_signal_smr1 = Channel_signal_output[0]
Channel_signal_smr1['section'] = np.digitize(Channel_signal_smr1.Time,juice_timestamp) # seperate analog signal by juice timestamps
# Remove tail of analog data
Channel_signal_smr1 = Channel_signal_smr1[Channel_signal_smr1['section']<Channel_signal_smr1['section'].unique()[-1]]
Channel_signal_smr1['Time'].iloc[-1] = juice_timestamp[-1]
# Remove head of analog data
Channel_signal_smr1 = Channel_signal_smr1[Channel_signal_smr1['Time']>marker_list[0]['values'][marker_list[0]['labels']==1][0]]


monkey_smr_dataframe = Channel_signal_smr1[["Time", "Signal stream 1", "Signal stream 2", "Signal stream 3", "Signal stream 10"]].reset_index(drop=True)
monkey_smr_dataframe.columns = ['monkey_t', 'monkey_x', 'monkey_y', 'monkey_speed', 'AngularV']
monkey_smr = dict(zip(monkey_smr_dataframe.columns.tolist(), np.array(monkey_smr_dataframe.values.T.tolist())))

"""### Data from monkey

ff & monkey information

speed of monkey = delta_position/t
delta_position = sqrt(delta_x^2 + delta_y^2)

Here we can only assume that the trajectory between two time points 
is straight.

To calculate speed_i, we use the information at t_i and t_i+1
"""

# for monkey
NEW_DATASET = False
MONKEY_DATA = True
NO_PLOT_NEEDED = True
data_folder_name = "0219"
data_num = 19

ff_information,monkey_information = log_extractor().extract_data()

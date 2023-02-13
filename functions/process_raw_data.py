import os
import math
import re
import os.path
import neo
import numpy as np
import pandas as pd
from math import pi
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class smr_extractor(object):   
    def __init__(self, data_folder_name):
        self.folder_path = data_folder_name
        self.files_names = [file for file in os.listdir(self.folder_path) if 'smr' in file]
        self.full_path_file_names = [os.path.join(self.folder_path,self.files_names[i]) 
                                     for i, value in enumerate(self.files_names)] # a list contains 2 file path in total
    
    def extract_data(self):
        Channel_signal_output = []
        marker_list = []
        
        for index, file_name in enumerate(self.full_path_file_names):  # loop 2 files one by one
            seg_reader = neo.io.Spike2IO(filename=file_name).read_segment() # read file
            
            if index == 0: # get sampling rate, only need to get it once
                smr_sampling_rate = seg_reader.analogsignals[0].sampling_rate 
                
            analog_length = min([i.size for i in seg_reader.analogsignals]) # in case analog channels have different shape
            
            Channel_index = [] # create an empty list to store channel names

            for C_index, C_data in enumerate(seg_reader.analogsignals[:-1]): # -1 indicates we disgard 'Raw' channel   
                shape = seg_reader.analogsignals[C_index].shape[1] # See how many channels are contained in each element of the list
                if C_index==0:
                  Channel_signal = C_data.as_array()[:analog_length,];
                else:
                  Channel_signal = np.append(Channel_signal, C_data.as_array()[:analog_length,], axis=1)
                for i in range(shape):
                  Channel_index.append(seg_reader.analogsignals[C_index].name) # get channel name one by one and put in Channel_index

            Channel_signal = np.append(Channel_signal, np.asarray(seg_reader.analogsignals[0].times[:analog_length,]).reshape(analog_length,1), axis=1)# get time stamps and put in Channel_signal
            Channel_index.append('Time') 

            Channel_signal_output.append(pd.DataFrame(Channel_signal,columns=Channel_index))

            marker_channel_index = [index for index,value in enumerate(seg_reader.events) if value.name == 'marker'][0] #find 'marker' channel
            marker_labels = seg_reader.events[marker_channel_index].get_labels().astype('int') # get 'marker' labels
            marker_values = seg_reader.events[marker_channel_index].as_array() # get 'marker' values
            marker = {'labels': marker_labels, 'values': marker_values} # arrange labels and values in a dict
            marker_list.append(marker)
            
        return Channel_signal_output, marker_list, smr_sampling_rate



# Read log file here
class log_extractor(object):
    def __init__(self,data_folder_name,file_name):
        self.folder_path = data_folder_name
        #self.files_names = [file for file in os.listdir(self.folder_path) if 'txt' and '0' in file]
        self.files_names = file_name
        #self.full_path_file_names = os.path.join(self.folder_path,self.files_names[0])
        self.full_path_file_names = os.path.join(self.folder_path,self.files_names)
        
    def extract_data(self):
        ffLinenumberList = []
        ff_information = []
        ffname_index = 0
        
        with open(self.full_path_file_names,'r',encoding='UTF-8') as content:
            log_content = content.readlines()
         
        for LineNumber, line in enumerate(log_content):
            key_ff = re.search('Firefly', line)
            key_monkey = re.search('Monkey', line)
            if key_ff is not None:
                ffLinenumberList.append(LineNumber)
            if key_monkey is not None:
                monkeyLineNum = LineNumber

        # get ff data
        for index, LineNumber in enumerate(ffLinenumberList):
            FF_Catched_T = []
            FF_Position = []
            FF_believed_position = []
            FF_flash_T = []
            
            if index == len(ffLinenumberList)-1:
                log_content_block = log_content[ffLinenumberList[index]+1:monkeyLineNum]
            else:
                log_content_block = log_content[ffLinenumberList[index]+1:ffLinenumberList[index+1]]
            
            for line in log_content_block:
                if len(line.split(' ')) == 5:
                    if 'inf' not in line:
                        FF_Catched_T.append(float(line.split(' ')[0]))
                        FF_Position.append([float(line.split(' ')[1]),float(line.split(' ')[2])])
                        FF_believed_position.append([float(line.split(' ')[3]),float(line.split(' ')[4])])
                else:
                    try:
                        FF_flash_T.append([float(line.split(' ')[0]),float(line.split(' ')[1])])
                    except:
                        1
                        
            FF_flash_T = np.array(FF_flash_T)
            seperated_ff = np.digitize(FF_flash_T.T[0],FF_Catched_T)
            for j in np.unique(seperated_ff)[:-1]:
                
                ff_information.append({'ff_index':ffname_index,'ff_catched_T':FF_Catched_T[j],'ff_real_position': np.array(FF_Position[j]),
                                  'ff_believed_position': np.array(FF_believed_position[j]),
                                      'ff_flash_T': FF_flash_T[seperated_ff==j]})  
                
                ffname_index = ffname_index+1    
                
        # get monkey data
        Monkey_X = []
        Monkey_Y = []
        Monkey_Position_T = []
        
        for line in log_content[monkeyLineNum+1:]:
            Monkey_X.append(float(line.split(' ')[0]))
            Monkey_Y.append(float(line.split(' ')[1]))
            Monkey_Position_T.append(float(line.split(' ')[2]))
            
        monkey_information = {'monkey_x': np.array(Monkey_X), 'monkey_y': np.array(Monkey_Y), 'monkey_t': np.array(Monkey_Position_T)}
        
        return ff_information, monkey_information
        #self.ff_information = ff_information
        #self.monkey_information = monkey_information



def process_monkey_information(monkey_information):
    """
    Get linear speeds, angular speeds, and angles of the monkey based on time, x-coordinates, and y-coordinates.

    Parameters
    ----------
    monkey_information: dict
        containing the time, x-coordinates, and y-coordinates of the monkey

    Returns
    -------
    monkey_information: dict
        containing more information of the monkey

    """

    delta_time = np.diff(monkey_information['monkey_t'])
    delta_x = np.diff(monkey_information['monkey_x'])
    delta_y = np.diff(monkey_information['monkey_y'])
    delta_position = np.sqrt(np.square(delta_x)+np.square(delta_y))
    monkey_speed = np.divide(delta_position, delta_time)
    monkey_speed = np.append(monkey_speed[0], monkey_speed)

    # If the monkey's speed at one point exceeds 200, we replace it with the previous speed.
    # (This can happen when the monkey reaches the boundary and comes out at another place)

    while np.where(monkey_speed>=200)[0].size > 0:
        index = np.where(monkey_speed>=200)[0]
        monkey_speed1 = np.append(np.array([0]), monkey_speed)
        monkey_speed[index] = monkey_speed1[index]
    monkey_information['monkey_speed'] = monkey_speed
    monkey_information['monkey_speeddummy'] = (monkey_information['monkey_speed']> 0.01).astype(int) 


    # Add angle of the monkey
    angle = [90]  # The monkey is at 90 degree angle at the beginning
    current_angle = 90 # This keeps track of the current angle during the iterations
    # Find the time in the data that is closest (right before) the time where we wan to know the monkey's angular position.
    for i in range(1, len(monkey_information['monkey_t'])):
      # If the monkey basically stopped at this moment, we keep the previous angle
      if monkey_information['monkey_speed'][i-1] < 1:  # use i-1 because 'monkey_speed' has 1 less element
        angle.append(current_angle)   
      else:
        # calculate the angle defined by two points
        myradians = math.atan2(monkey_information['monkey_y'][i]-monkey_information['monkey_y'][i-1], 
                               monkey_information['monkey_x'][i]-monkey_information['monkey_x'][i-1])
        mydegrees = math.degrees(myradians)

        # Compare the new angle with the previous angle:
        # If the difference is larger than that can be allowed by the angular speed, 
        # then the monkey might be going backward, and we will just subtract 180 from the angle
        if abs(mydegrees-current_angle) % 360 > 20:
        # if 170 < abs(mydegrees-current_angle)%360 < 190: # This is the original code
            current_angle = current_angle - 180
            if current_angle < -180:
                current_angle += 360
        # else, we keep the new angle
        else: 
            current_angle = mydegrees
        angle.append(current_angle)
    angle = np.array(angle)
    monkey_information['monkey_angle'] = angle*pi/180

    delta_angle = np.diff(monkey_information['monkey_angle'])
    delta_angle = np.remainder(delta_angle, 2*pi)
    monkey_dw = np.divide(delta_angle, delta_time)
    monkey_dw = np.append(monkey_dw[0], monkey_dw)
    monkey_information['monkey_dw'] = monkey_dw
    return monkey_information




def unpack_ff_information_of_monkey(raw_data_folder_name, ff_information):
    """
    Get various useful lists and arrays of ff information from the raw data

    Parameters
    ----------
    raw_data_folder_name: str
        the folder name of the raw data
    ff_information: list
        derived from the raw data; note that it is different from env.ff_information


    Returns
    -------
    ff_catched_T_sorted: np.array
        containing the time when each captured firefly gets captured
    ff_index_sorted: np.array
        containing the sorted indices of the fireflies
    ff_real_position_sorted: np.array
        containing the real locations of the fireflies
    ff_believed_position_sorted: np.array
        containing the locations of the monkey (or agent) when each captured firefly was captured 
    ff_life_sorted: np.array
        containing the time that each firefly comes into being and gets captured 
        (if the firefly is never captured, then capture time is replaced by the last point of time in data)
    ff_flash_sorted: np.array
        containing the flashing-on durations of each firefly 
    ff_flash_end_sorted: np.array
        containing the end of each flash-on duration of each firefly

    """

    Channel_signal_output, marker_list, smr_sampling_rate = smr_extractor(data_folder_name = raw_data_folder_name).extract_data()
    # Considering the first smr file, use marker_list[0], Channel_signal_output[0]
    juice_timestamp = marker_list[0]['values'][marker_list[0]['labels'] == 4]
    Channel_signal_smr1 = Channel_signal_output[0]
    Channel_signal_smr1['section'] = np.digitize(Channel_signal_smr1.Time, juice_timestamp) # seperate analog signal by juice timestamps
    # Remove tail of analog data
    Channel_signal_smr1 = Channel_signal_smr1[Channel_signal_smr1['section']<  Channel_signal_smr1['section'].unique()[-1]]
    Channel_signal_smr1.loc[Channel_signal_smr1.index[-1], 'Time'] = juice_timestamp[-1]
    # Remove head of analog data
    Channel_signal_smr1 = Channel_signal_smr1[Channel_signal_smr1['Time'] > marker_list[0]['values'][marker_list[0]['labels']==1][0]]
    # monkey_smr_dataframe = Channel_signal_smr1[["Time", "Signal stream 1", "Signal stream 2", "Signal stream 3", "Signal stream 10"]].reset_index(drop=True)
    # monkey_smr_dataframe.columns = ['monkey_t', 'monkey_x', 'monkey_y', 'monkey_speed', 'AngularV']
    # monkey_smr = dict(zip(monkey_smr_dataframe.columns.tolist(), np.array(monkey_smr_dataframe.values.T.tolist())))


    ff_index = []
    ff_catched_T = []
    ff_real_position = []
    ff_believed_position = []
    ff_life = []
    ff_flash = []
    ff_flash_end = []  # This is the time that the firefly last stops flash
    for item in ff_information:
        item['Life'] = np.array([item['ff_flash_T'][0][0], item['ff_catched_T']])
        ff_index = np.hstack((ff_index, item['ff_index']))
        ff_catched_T = np.hstack((ff_catched_T, item['ff_catched_T']))
        ff_real_position.append(item['ff_real_position'])
        ff_believed_position.append(item['ff_believed_position'])
        ff_life.append(item['Life'])
        ff_flash.append(item['ff_flash_T'])
        ff_flash_end.append(item['ff_flash_T'][-1][-1])
    sort_index = np.argsort(ff_catched_T)
    ff_catched_T_sorted = ff_catched_T[sort_index]
    # Use accurate juice timestamps, ff_catched_T_sorted for smr1 (so that the time frame is correct)
    last_valid_point = np.where(ff_catched_T_sorted <= Channel_signal_smr1.Time.values[-1])[0][-1]+1
    ff_catched_T_sorted = ff_catched_T_sorted[:last_valid_point]
    ff_index_sorted = ff_index[sort_index][:last_valid_point]
    ff_real_position_sorted = np.array(ff_real_position)[sort_index][:last_valid_point]
    ff_believed_position_sorted = np.array(ff_believed_position)[sort_index][:last_valid_point]
    ff_life_sorted = np.array(ff_life)[sort_index][:last_valid_point]
    ff_flash_sorted = np.array(ff_flash, dtype=object)[sort_index][:last_valid_point].tolist()
    ff_flash_end_sorted = np.array(ff_flash_end)[sort_index][:last_valid_point]
    return ff_catched_T_sorted, ff_index_sorted, ff_real_position_sorted, ff_believed_position_sorted, ff_life_sorted, ff_flash_sorted, ff_flash_end_sorted


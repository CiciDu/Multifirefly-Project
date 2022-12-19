import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import math
from math import pi
import re
import os
import os.path
import neo
import numpy as np
import pandas as pd


class smr_extractor(object):   
    def __init__(self, data_folder_name):
        self.folder_path = data_folder_name
        self.files_names = [file for file in os.listdir(self.folder_path) if 'smr' in file]
        self.full_path_file_names = [os.path.join(self.folder_path,self.files_names[i]) 
                                     for i,value in enumerate(self.files_names)] # a list contains 2 file path in total
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
                
        #get monkey data
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
    delta_time = np.diff(monkey_information['monkey_t'])
    delta_x = np.diff(monkey_information['monkey_x'])
    delta_y = np.diff(monkey_information['monkey_y'])
    delta_position = np.sqrt(np.square(delta_x)+np.square(delta_y))
    monkey_speed = np.divide(delta_position, delta_time)
    monkey_speed = np.append(np.array([0]), monkey_speed)

    # If the monkey's speed at one point exceeds 200 
    #(this can happen when the monkey reaches the boundary and comes out at another place)
    # we replace it with the previous speed.
    while np.where(monkey_speed>=200)[0].size > 0:
      index = np.where(monkey_speed>=200)[0]
      monkey_speed1 = np.append(np.array([0]), monkey_speed)
      monkey_speed[index] = monkey_speed1[index]
    monkey_information['monkey_speed'] = monkey_speed
    monkey_information['monkey_speeddummy'] = (monkey_information['monkey_speed']> 0.01).astype(int) 
    monkey_speeddummy = (monkey_speed>0.1).astype(int) 


    # Add angle of the monkey

    angle = [90]  # The monkey is at 90 degree angle at the beginning
    current_angle = 90 # This keeps track of the current angle during the iterations
    # Find the time in the data that is closest (right before) the time where we wan to know the monkey's angular position.
    for i in range(1, len(monkey_information['monkey_t'])):
      # If the monkey basically stopped at this moment, we keep the previous angle
      if monkey_information['monkey_speed'][i-1] < 1:  # use i-1 because 'monkey_speed' has 1 less element
        angle.append(current_angle)   
      else:
        myradians = math.atan2(monkey_information['monkey_y'][i]-monkey_information['monkey_y'][i-1], monkey_information['monkey_x'][i]-monkey_information['monkey_x'][i-1])
        mydegrees = math.degrees(myradians)
        # Compare the new angle with the previous angle
        # If the different is too large, then the monkey might be going backward, and we will just subtract 180 from the angle
        if 170 < abs(mydegrees-current_angle)%360 < 190 :
          current_angle = current_angle - 180
          if current_angle < -180:
            current_angle += 360
          angle.append(current_angle)
        else: #else, we keep the new angle
          current_angle = mydegrees
          angle.append(current_angle)
    angle = np.array(angle)
    monkey_information['monkey_angle'] = angle*pi/180

    delta_time = np.delete(monkey_information['monkey_t'], 0) - np.delete(monkey_information['monkey_t'], -1)
    delta_angle = np.delete(monkey_information['monkey_angle'], 0) - np.delete(monkey_information['monkey_angle'], -1)
    monkey_dw = np.append(np.array([0]), np.divide(delta_angle, delta_time))
    monkey_information['monkey_dw'] = monkey_dw

    monkey_dw = np.divide(np.diff(np.array(monkey_information['monkey_angle']), prepend=monkey_information['monkey_angle'][0]), np.append(delta_time[0], delta_time))
    monkey_information['monkey_dw'] = monkey_dw
    return monkey_information

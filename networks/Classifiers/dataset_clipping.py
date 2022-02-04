# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from itertools import chain

# Data params
textures = 11
trials = 100

# Set bin and sim params
bin_size = 50   # Bin size in ms
sim_length = 5000   # Sim time in ms
bins = sim_length / bin_size

# Function to remove repeated spikes from data
# New numpy based function
def remove_duplicates(data):
    new_data = []
    
    for i in range(len(data)):
        new_data.append(np.unique(data[i]))
        
    return new_data
    
# Function to remove empty lists from nested list
def rmv_empty(data):
    # Remove empty lists
    new_list = [x for x in data if x != []]
    return new_list
    
# Function to find the starting point of spiking events
def find_start(data):
    # Convert nested lists into single list
    new_list = chain.from_iterable(data)
    
    # Remove empty lists
    new_list = rmv_empty(new_list)
    
    # Order new list by value
    new_list = np.array(new_list)
    sorted_list = np.sort(new_list)
    
    # Find starting point
    count = 0
    binny = 5   # Steps in ms
    threshold = 30   # Threshold for delcaring the starting point of activity
    
    # Loop through array with "binny" sized steps
    for j in range(0, np.max(sorted_list), binny):
        
        # Count how many spikes are within this bin
        for p in range(len(sorted_list)):
            if (sorted_list[p] > j) & (sorted_list[p] < j + binny):
                count = count +1
                
        # If the count within this bin reaches a threshold, its declared as the starting value
        if count > threshold:
            return j
        else:
            count = 0
    
    
# Function to remove spikes before a certain time from dataset
def rmv_start(data, start):
    temp_data = []
    new_data = []
    
    # Loop through dataset
    for l in range(len(data)):
        # Loop through nested loop
        for j in data[l]:
            # If a spike occured after the start time then append
            if j > start:
                temp_data.append(j)
        
        # Append temp list to the new dataset and clear temp list
        new_data.append(temp_data)
        temp_data = []
    
    return new_data
    
# Import and bin data
dataset = np.empty([int(bins)])


for kk in range(textures):
    for ll in range(trials):
        # Path to file containing spike timings
        FILE_NAME = "Artificial Dataset " + str(ll) + "Texture No. " + str(kk) + ".pickle"
        DATA_PATH = "/home/farscope2/Documents/PhD/Spiking Nets Project/SpikingNetsTexture/datasets/TacTip_NM/Reduced/" + FILE_NAME

        # Import and flatten the dataset for use in the network
        spike_times = np.load(DATA_PATH, allow_pickle=True)
        spks = spike_times.reshape(-1)

        
        # Move through array and remove data points upto the point of first action
        clipped_spks = rmv_start(spks, find_start(spks))        

        # Remove duplicates using previous function
        clean_spks = remove_duplicates(clipped_spks)
        
        # List to hold all data from item
        data = []
        value = 0

        # Loop untill all bins are full
        for bin_ms in range(find_start(spks), find_start(spks) + sim_length, bin_size):
            # Loop entire population
            for t in range(len(clean_spks)):
                # Check number of times each neuron spiked within this timeframe
                ar = np.array(clean_spks[t])
                value += np.count_nonzero(((ar >= bin_ms) & (ar < bin_ms + bin_size)))

            data.append(value)
            value = 0
        
        # DEBUG
        print("Trial: " + str(ll) + "Texture: " + str(kk))
        #print(dataset)
        
        # If dataset is currently empty then dataset is formed from data list
        # Else create row below current dataset to input new data
        if dataset.size == 0:
            dataset[0] = np.array(data)
        else:
            dataset = np.vstack([dataset, data])
    
    
# Pickle the dataset for ease of future use
# This part of the notebook need only be used once

data_dir = "/home/farscope2/Documents/PhD/Spiking Nets Project/SpikingNetsTexture/datasets/TacTip_NM/300ms clipped"
pickle_out = open(os.path.join(data_dir, str(sim_length) + "ms - " + str(bin_size) + "ms bin size dataset.pickle"), 'wb')
pickle.dump(dataset, pickle_out)
pickle_out.close()

# pickle_out = open(os.path.join(data_dir, str(textures) + " textures - " + str(trials) + " trials labels.pickle"), 'wb')
# pickle.dump(labels, pickle_out)
# pickle_out.close()

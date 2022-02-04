# Import libraries
# import nest
# import pyNN.nest as sim
# from pyNN.parameters import Sequence
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.defchararray import array
import math
from itertools import chain
#import time

# Function to remove repeated spikes from data
# New numpy based function


def remove_duplicates(data):
    new_data = []

    for i in range(len(data)):
        new_data.append(np.unique(data[i]))

    return new_data

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
                count = count + 1

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

# Function to remove empty lists from nested list


def rmv_empty(data):
    # Remove empty lists
    new_list = [x for x in data if x != []]
    return new_list


# Function to set appropriate figure size for publication
# Width @516pts is for IEEE conference format
def set_size(width=516, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


for ll in range(1):
    for kk in range(0, 11):
        # Reset NEST kernal
        # nest.ResetKernel()  # Reset kernal to prevent errors

        # Setup connection from pyNN to NEST backend
        # sim.setup(timestep=1.0)

        # Path to file containing spike timings
        FILE_NAME = "Artificial Dataset " + \
            str(ll) + "Texture No. " + str(kk) + ".pickle"
        DATA_PATH = "/home/farscope2/Documents/PhD/Spiking Nets Project/SpikingNetsTexture/datasets/TacTip_NM/Reduced/" + FILE_NAME

        SAVE_LOCATION = "/home/farscope2/Documents/PhD/Spiking Nets Project/SpikingNetsTexture/graphs/clipped_hist_long/"

        # Import and flatten the dataset for use in the network
        spike_times = np.load(DATA_PATH, allow_pickle=True)
        spks = spike_times.reshape(-1)
        # print(spks)

        # Move through array and remove data points upto the point of first action
        clipped_spks = rmv_start(spks, find_start(spks))

        # Remove duplicates using previous function
        clean_spks = remove_duplicates(clipped_spks)

        # Number used to track current bin levels
        data = []
        value = 0
        bin_ms = 0

        # # Loop while the end of data hasn't been reached
        # for bin_ms in range(0, 300, 10):
        #     # Loop entire population
        #     for t in range(len(spks)):
        #         # Check number of times each neuron spiked within this timeframe
        #         ar = np.array(spks[t])
        #         value += np.count_nonzero(((ar >= bin_ms) & (ar < bin_ms+10)))

        #     data.append(value)
        #     value = 0
        #     #bin_ms += 10

        # Loop through data
        for sublist in clean_spks:
            # Loop through sublist
            for item in sublist:
                data.append(item)

        # print(data)
        data.sort()
        data = np.array(data)
        print(data)
        w = 10
        n = math.ceil((data.max() - data.min())/w)
        print(n)
        # print(len(data))
        # width = 0.8
        # y = np.arange(0, len(data)*10, step=10)
        # plt.bar(y, data, width)

        # Plot histogram from data
        plt.rcParams["figure.figsize"] = set_size(fraction=0.5)
        #plt.hist(data, bins = 30, density=True)
        # range(0,len(data),10)    # bins = 10
        plt.hist(data, bins=n, density=True)
        plt.xlabel("Time (ms)")
        plt.ylabel("Spike Density")
        # plt.title(FILE_NAME)
        plt.xlim(find_start(spks), find_start(spks) + 5001)
        plt.yticks([])
        #plt.xticks(np.arange(0, 300, step=10))
        plt.grid(True)
        plt.savefig(SAVE_LOCATION + FILE_NAME + "ms.eps", bbox_inches="tight")
        plt.clf()
        plt.show()

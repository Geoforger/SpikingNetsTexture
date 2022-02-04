# Import libraries
# import nest
# import pyNN.nest as sim
# from pyNN.parameters import Sequence
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.defchararray import array
import math
from itertools import chain

from numpy.core.fromnumeric import size
from numpy.lib.histograms import histogram
# import time

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


def euclidean_dist(x, y):
    """
    Calculate euclidean distance between two vectors
    """

    return np.sqrt(np.sum((x - y) ** 2))


# Main function of program
if __name__ == "__main__":

    # Hyperparameters
    trials = 50
    textures = 2
    testing_perc = 0.2

    peristims = []  # np.empty(textures)

    print("Creating peristimulus histograms")
    # Loop through each texture
    for ll in range(textures):

        larger_data = np.array([])

        # Loop through each training data point for said texture
        # Create a peristimulus histogram for the texture
        for kk in range(int(trials - (trials * testing_perc))):

            # Path to file containing spike timings
            FILE_NAME = "Artificial Dataset " + \
                str(kk) + "Texture No. " + str(ll) + ".pickle"
            DATA_PATH = "/home/farscope2/Documents/PhD/Spiking Nets Project/SpikingNetsTexture/datasets/TacTip_NM/Reduced_natural/" + FILE_NAME

            # SAVE_LOCATION = "/home/farscope2/Documents/PhD/Spiking Nets Project/SpikingNetsTexture/graphs/peri_hist_natural/"

            # Import and flatten the dataset for use in the network
            spike_times = np.load(DATA_PATH, allow_pickle=True)
            spks = spike_times.reshape(-1)

            larger_data = np.concatenate((larger_data, spks), axis=0)

        # print(larger_data)
        print("Texture " + str(ll))

        # Move through array and remove data points upto the point of first action
        # clipped_spks = rmv_start(spks, find_start(spks))
        clipped_spks = rmv_start(larger_data, find_start(larger_data))

        # Remove duplicates using previous function
        # clean_spks = remove_duplicates(spks)
        clean_spks = remove_duplicates(clipped_spks)

        # Number used to track current bin levels
        data = []
        value = 0
        bin_ms = 0

        # Loop through data
        for sublist in spks:  # clean_spks:
            # Loop through sublist
            for item in sublist:
                data.append(item)

        data.sort()
        data = np.array(data)
        # print(size(data))
        w = 10  # Size of bins
        n = math.ceil((data.max() - data.min())/w)
        # print(n)

        counts, bins = np.histogram(data, bins=500)
        # Scale the bins by the number of iterations
        counts = counts / trials

        # List of all the peristimulus histograms
        peristims.append(counts)  # [ll] = counts  # , bins

    print("Comparing peristimulus histograms")
    # Compare testing histograms to the peristimulus for each texture
    # Iterate through each texture
    for ll in range(textures):
        # Iterate through the testing data for each texture
        for kk in range(int(trials * testing_perc)):

            # Index to prevent training data from being included in testing
            data_index = trials - kk

            # Import and flatten the dataset for use in the network
            FILE_NAME = "Artificial Dataset " + \
                str(data_index) + "Texture No. " + str(ll) + ".pickle"

            # print(FILE_NAME)

            spike_times = np.load(DATA_PATH, allow_pickle=True)
            spks = spike_times.reshape(-1)

            # Move through array and remove data points upto the point of first action
            clipped_spks = rmv_start(spks, find_start(spks))

            # Remove duplicates using previous function
            clean_spks = remove_duplicates(clipped_spks)

            # Create histogram of the datapoint
            data = []

            # Loop through data
            for sublist in clean_spks:
                # Loop through sublist
                for item in sublist:
                    data.append(item)

            data.sort()
            data = np.array(data)
            # w = 10  # Size of bins
            # n = math.ceil((data.max() - data.min())/w)

            # Create histogram
            counts, bins = np.histogram(data, bins=500)
            #counts = counts / int(trials * testing_perc)

            # Compare histogram to the peristimulus for each texture
            euc_distances = np.empty([textures])

            print("Real texture = " + str(ll))
            for peri in range(textures):
                # Compare each texture peristimulus to the current testing histogram
                euc_distances[peri] = euclidean_dist(counts, peristims[peri])
                # euc_distances.append(np.linalg.norm(counts-peristims[peri]))
                print("Distance to texture " + str(peri) + " = " +
                      str(euclidean_dist(counts, peristims[peri])))

            min_value = np.argmin(euc_distances)
            #min_index = euc_distances.index(min_value)
            #min_index = np.where(euc_distances == min_value)
            print("Closest texture = " + str(min_value) + "\n")
            # euc_distances.clear()

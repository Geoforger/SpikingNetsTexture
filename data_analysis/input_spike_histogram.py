# Import libraries
# import nest
# import pyNN.nest as sim
# from pyNN.parameters import Sequence
import numpy
import matplotlib.pyplot as plt
from numpy.core.defchararray import array
import math
#import time

for kk in range(0, 11):
    # Reset NEST kernal
    #nest.ResetKernel()  # Reset kernal to prevent errors

    # Setup connection from pyNN to NEST backend
    #sim.setup(timestep=1.0)

    # Path to file containing spike timings
    FILE_NAME = "Artificial Dataset 0Texture No. " + str(kk) + ".pickle"
    DATA_PATH = "/home/farscope2/Documents/PhD/Spiking Nets Project/SpikingNetsTexture/datasets/TacTip_NM/Reduced/" + FILE_NAME
    
    SAVE_LOCATION = "/home/farscope2/Documents/PhD/Spiking Nets Project/SpikingNetsTexture/graphs/spike_hist_long/"

    # Define network
    #N = 19600   # Number of input neurons
    #run_time = 300.0

    # Import and flatten the dataset for use in the network
    spike_times = numpy.load(DATA_PATH, allow_pickle=True)
    spks = spike_times.reshape(-1)
    #print(spks)

    # Number used to track current bin levels
    data = []
    value = 0
    bin_ms = 0

    # # Loop while the end of data hasn't been reached
    # for bin_ms in range(0, 300, 10):
    #     # Loop entire population
    #     for t in range(len(spks)):
    #         # Check number of times each neuron spiked within this timeframe
    #         ar = numpy.array(spks[t])
    #         value += numpy.count_nonzero(((ar >= bin_ms) & (ar < bin_ms+10)))

    #     data.append(value)
    #     value = 0
    #     #bin_ms += 10

    # Loop through data
    for sublist in spks:
        # Loop through sublist
        for item in sublist:
            data.append(item)
    
    #print(data)
    data.sort()
    data = numpy.array(data)
    #print(data)
    w = 10
    n = math.ceil((data.max() - data.min())/w)

    #print(len(data))
    # width = 0.8
    # y = numpy.arange(0, len(data)*10, step=10)
    # plt.bar(y, data, width)

    # Plot histogram from data
    plt.rcParams["figure.figsize"] = (20,10)
    #plt.hist(data, bins = 30, density=True)
    plt.hist(data, bins = n, density=True)    #range(0,len(data),10)    # bins = 10
    plt.xlabel("Time (ms)")
    plt.ylabel("Spike Density")
    plt.title(FILE_NAME)
    plt.xlim(0, 5000)
    #plt.xticks(numpy.arange(0, 300, step=10))
    plt.grid(True)
    plt.savefig(SAVE_LOCATION + FILE_NAME + "ms.png")
    plt.clf()
    #plt.show()
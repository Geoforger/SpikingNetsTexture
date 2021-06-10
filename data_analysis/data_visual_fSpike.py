import pickle
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.arraysetops import unique

# Path to dataset
# Change folder path for each dataset?
FOLDER_NAME = "ntac_2.5_6texture_20trial_slide_test_06031504"
PATH = "/home/farscope2/Documents/PhD/Spiking Nets Project/SpikingNetsTexture/datasets/TacTip_NM/" + FOLDER_NAME

# Number of textures tested & number of trials per texture
textures = 6
trials = 20

# Maximum intensity of events seen across entire dataset
# Used to scale the heatmaps
max_time = 400
min_time = 0
#max_intensities = []

# Create array to contain the intensities of events for mapping
timings = np.zeros([240, 180])

# Open each file and find the highest intensity in order to find an appropriate vmax
for t in range(trials):
    for s in range(textures):
        # Open each file in dataset
        FILENAME = PATH + "/SingleTap_run_" + \
            str(t) + "_orientation_" + str(s) + ".pickle"

        with(open(FILENAME, "rb")) as openfile:
            try:
                orig_array = pickle.load(openfile)
            except EOFError:
                print(EOFError)

        for z in range(len(orig_array)):
            for y in range(len(orig_array[z])):
                timings[z, y] = min(orig_array[z, y], default=min_time)

        # Find earliest and latest times for spikes
        # earliest = np.min(timings)
        # lastest = np.max(timings)
        # # max_intensities.append(file_intensity)

        # if file_intensity > max_intensity:
        #     max_intensity = file_intensity

# print(np.unique(max_intensities))

# Create and save heatmap for each tap
for xx in range(trials):
    for yy in range(textures):
        # Open each file individually
        FILENAME = PATH + "/SingleTap_run_" + \
            str(xx) + "_orientation_" + str(yy) + ".pickle"

        # Create array of intensities for heatmap
        with(open(FILENAME, "rb")) as openfile:
            try:
                orig_array = pickle.load(openfile)
            except EOFError:
                print(EOFError)

        for u in range(len(orig_array)):
            for i in range(len(orig_array[u])):
                timings[u, i] = min(orig_array[u, i], default=min_time)

        # Plot heatmap of events
        plt.imshow(timings, cmap='hot', interpolation='nearest',
                   vmin=min_time, vmax=max_time, label="Timing of first spike (ms)")
        plt.ylabel('Y Pixels')
        plt.xlabel('X Pixels')
        plt.colorbar()
        plt.title("SingleTap_run_" +
                  str(xx) + "_orientation_" + str(yy) + " spiking locations over time")
        plt.savefig("/home/farscope2/Documents/PhD/Spiking Nets Project/SpikingNetsTexture/first_spike_graphs/SingleTap_run_" +
                    str(xx) + "_orientation_" + str(yy) + ".pickle" + ".png")
        plt.clf()  # Clear figure post save
        # plt.show()

import pickle
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.arraysetops import unique

# Path to dataset
# Change folder path for each dataset?
FOLDER_NAME = "ntac_2.5_11texture_20trial_slide_test_06101055/"
PATH = "/home/farscope2/Documents/PhD/Spiking Nets Project/SpikingNetsTexture/datasets/TacTip_NM/" + FOLDER_NAME

# Number of textures tested & number of trials per texture
textures = 11
trials = 20

# Max intensity of events seen across each texture within dataset
t1 = []
t2 = []
t3 = []
t4 = []
t5 = []
t6 = []
t7 = []
t8 = []
t9 = []
t10 = []
t11 = []

no_spikes = 0

# Create and save heatmap for each tap
for xx in range(trials):
    for yy in range(textures):
        # Open each file individually
        FILENAME = PATH + "Artificial Dataset " + \
            str(xx) + "Texture No. " + str(yy) + ".pickle"

        # Create array of intensities for heatmap
        with(open(FILENAME, "rb")) as openfile:
            try:
                orig_array = pickle.load(openfile)
            except EOFError:
                print(EOFError)

        # Find maximum intensity for this datapoint
        for u in range(len(orig_array)):
            for i in range(len(orig_array[u])):
                # Check if lits is empty
                if orig_array[u, i]:
                    # If list is populated, add 1 to count
                    no_spikes += 1

        # Append the maximum spike intensity from the datapoint to its array
        if yy == 0:
            t1.append(no_spikes)
        elif yy == 1:
            t2.append(no_spikes)
        elif yy == 2:
            t3.append(no_spikes)
        elif yy == 3:
            t4.append(no_spikes)
        elif yy == 4:
            t5.append(no_spikes)
        elif yy == 5:
            t6.append(no_spikes)
        elif yy == 6:
            t7.append(no_spikes)
        elif yy == 7:
            t8.append(no_spikes)
        elif yy == 8:
            t9.append(no_spikes)
        elif yy == 9:
            t10.append(no_spikes)
        elif yy == 10:
            t11.append(no_spikes)

        # Reset maximum count for next iteration
        no_spikes = 0


# X_vals are textures
x_vals = list(range(0, textures, 1))

y_vals = [np.mean(t1), np.mean(t2), np.mean(t3),
          np.mean(t4), np.mean(t5), np.mean(t6), np.mean(t7), np.mean(t8), np.mean(t9), np.mean(t10), np.mean(t11)]

# Create bar plot
plt.bar(x_vals, y_vals, color='green')
plt.xlabel("Texture Number")
plt.ylabel("Number of unique spiking events")
plt.plot()
plt.show()

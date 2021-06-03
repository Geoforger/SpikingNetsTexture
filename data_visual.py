import pickle
import matplotlib.pyplot as plt
import numpy as np

# Create array to contain the intensities of
intensity = np.zeros([240, 180])
FOLDER_NAME = "ntac_2.5_6texture_20trial_slide_test_06031504"
PATH = "/Documents/PhD/Spiking Nets Project/SpikingNetsTexture/datasets/TacTip_NM/" + FOLDER_NAME

with(open("SingleTap_run_0_orientation_1.pickle", "rb")) as openfile:
    try:
        orig_array = pickle.load(openfile)
    except EOFError:
        print(EOFError)

for z in range(len(orig_array)):
    for y in range(len(orig_array[z])):
        intensity[z, y] = len(orig_array[z, y])


# Plot heatmap of events
plt.imshow(intensity, cmap='hot', interpolation='nearest', vmin=0, vmax=22)
plt.ylabel('Y Pixels')
plt.xlabel('X Pixels')
plt.colorbar()
plt.savefig("SingleTap_run_0_orientation_1.png")
plt.show()

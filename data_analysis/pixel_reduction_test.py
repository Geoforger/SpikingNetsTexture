import pickle
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.arraysetops import unique
import os

# Path to dataset
# Change folder path for each dataset
FOLDER_NAME = "ntac_2.5_11texture_100trial_slide_test_06101340/"
PATH = "/home/farscope2/Documents/PhD/Spiking Nets Project/SpikingNetsTexture/datasets/TacTip_NM/" + FOLDER_NAME
# Path to save pickle data in
data_dir = "/home/farscope2/Documents/PhD/Spiking Nets Project/SpikingNetsTexture/datasets/TacTip_NM/Reduced"


# Number of textures tested & number of trials per texture
textures = 11
trials = 100

# Maximum intensity of events seen across entire dataset
# Used to scale the heatmaps
max_intensity = 50
max_intensities = []

# Image reduction parameters
x_red = 40  # Total horizontal crop of image
temp_y = 50 # The crop from the top of the image
y_red = temp_y + 40  # The total vertical crop of the image

# Create array to contain the intensities of events for mapping
# Create array to contain only the pixels of the neurotac tip
intensity = np.zeros([240-y_red,180-x_red])
temp_image = np.empty([240 - temp_y,180-x_red], dtype=object)  # Temp image removes missing pixels at top
reduced_image = np.empty([240-y_red,180-x_red], dtype=object)


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

        # Remove excess pixels from the top and sides of image
        for c in range(len(temp_image)):
            for d in range(len(temp_image[0])):
                temp_image[c, d] = orig_array[c+temp_y, d+x_red] #c+y_red

        # Remove the excess bottom pixels
        for y in range(len(reduced_image)):
            for z in range(len(reduced_image[0])):
                reduced_image[y,z] = temp_image[y,z]

        for u in range(len(reduced_image)):
                for i in range(len(reduced_image[u])):
                    intensity[u, i] = len(reduced_image[u, i])


        # Save dataset seperately for ease of use with network
        pickle_out = open(os.path.join(data_dir, 'Artificial Dataset ' +
                                               str(xx) + 'Texture No. ' + str(yy) + '.pickle'), 'wb')
        pickle.dump(reduced_image, pickle_out)
        pickle_out.close()

        # Plot heatmap of events
        plt.imshow(intensity, cmap='Reds', interpolation='nearest',
                vmin=0, vmax=max_intensity)
        plt.ylabel('Y Pixels')
        plt.xlabel('X Pixels')
        plt.colorbar()
        plt.title("Artificial Dataset Trial " +
                str(xx) + " Texture No. " + str(yy) + " Event Intensity")
        plt.savefig("/home/farscope2/Documents/PhD/Spiking Nets Project/SpikingNetsTexture/graphs/reduced/" +
                    FOLDER_NAME + "Artificial Dataset " + str(xx) + " Texture" + str(yy) + ".pickle" + ".png")
        plt.clf()  # Clear figure post save
        # plt.show()
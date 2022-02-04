import pickle
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.arraysetops import unique
import os


def pixel_reduction(data, x_reduce, y_reduce, x_ratio=0.5, y_ratio=0.5):
    """ Function to reduce the number of pixels in output image from the neuroTac

    Arguments
    ----------
    data:   nested list (array of lists)
                    Nested array of timestamps that requires a reduction in size
    x_reduce:   int
                    Total number of pixels to reduce along the x axis
    y_reduce:   int
                    Total number of pixels to reduce along the y axis
    x_ratio:    float
                    Ratio of pixels to remove from left to right on x axis. Ie. 0.5 gives x_reduce * 0.5 pixels removed from left and right of image (default = 0.5)
    y_ratio:    float
                    Ratio of pixels to remove from top to bottom on y axis. Ie. 0.5 gives y_reduce * 0.5 pixels removed from top and bottom of image (default = 0.5)
    Returns
    -------
    reduced_image:  nested list (array of lists)
                        New cropped array of timestamps
    """

    # Find the size of the array being reduced
    x_size = data.shape[1]
    y_size = data.shape[0]

    # print(x_size)
    # print(y_size)

    # Find number of pixels to crop from each side
    # x_crop_left = int(x_reduce * (1 - x_ratio))
    # x_crop_right = int(x_reduce - x_crop_left)
    # y_crop_top = int(y_reduce * (1 - y_ratio))
    # y_crop_bottom = int(y_reduce - y_crop_top)
    x_boundary_left = int(x_reduce * x_ratio)
    x_boundary_right = int(x_size - (x_reduce * (1-x_ratio)))
    y_boundary_top = int(y_reduce * y_ratio)
    y_boundary_bottom = int(y_size - (y_reduce * (1 - y_ratio)))

    # print(f"x_boundary_left = {x_boundary_left}")
    # print(f"x_boundary_right = {x_boundary_right}")
    # print(f"y_boundary_top = {y_boundary_top}")
    # print(f"y_boundary_bottom = {y_boundary_bottom}")

    # Create an empty reduced image array to store our data into
    # dtype=object as each element of the array will contain a list
    reduced_image = np.zeros(
        shape=((y_size - y_reduce), (x_size - x_reduce)), dtype=object)

    # print(reduced_image.shape)

    # Cycle through the original array
    for x in range(x_size):
        for y in range(y_size):
            # print(f"{x},{y}")
            # If image is within the boundaries of the new image
            if (x_boundary_left <= x < x_boundary_right) and (y_boundary_top <= y < y_boundary_bottom):
                reduced_image[y-y_boundary_top, x-x_boundary_left] = data[y, x]

    return reduced_image


# Path to dataset
# Change folder path for each dataset
FOLDER_NAME = "ntac_2.5_11texture_100trial_slide_test_06101340/"
PATH = "/home/farscope2/Documents/PhD/First_Year_Project/SpikingNetsTexture/datasets/TacTip_NM/" + FOLDER_NAME
# Path to save pickle data in
data_dir = "/home/farscope2/Documents/PhD/First_Year_Project/SpikingNetsTexture/datasets/TacTip_NM/Reduced_natural_0/"


# Number of textures tested & number of trials per texture
textures = 11
trials = 100

# Maximum intensity of events seen across entire dataset
# Used to scale the heatmaps
max_intensity = 50
max_intensities = []

# Image reduction parameters
x_red = 40  # Total horizontal crop of image
temp_y = 50  # The crop from the top of the image
y_red = temp_y + 40  # The total vertical crop of the image

# Create array to contain the intensities of events for mapping
# Create array to contain only the pixels of the neurotac tip
#intensity = np.zeros([240-y_red, 180-x_red])
# Temp image removes missing pixels at top
#temp_image = np.empty([240 - temp_y, 180-x_red], dtype=object)
#reduced_image = np.empty([240-y_red, 180-x_red], dtype=object)

FILENAME = PATH + "Artificial Dataset 1Texture No. 1.pickle"

# Create array of intensities for heatmap
with(open(FILENAME, "rb")) as openfile:
    try:
        orig_array = pickle.load(openfile)
    except EOFError:
        print(EOFError)

new_image = pixel_reduction(
    orig_array, 40, 90, 1.0, 0.55)
print(new_image)

intensity = np.zeros_like(new_image, dtype=int)
print(intensity)

for x in range(new_image.shape[1]):
    for y in range(new_image.shape[1]):
        intensity[y, x] = len(new_image[y, x])

# Plot heatmap of events
plt.imshow(intensity, interpolation='nearest',
           vmin=0, vmax=max_intensity)  # cmap='Reds'
plt.ylabel('Y Pixels')
plt.xlabel('X Pixels')
plt.colorbar()
plt.title("Artificial Dataset 1Texture No. 1.pickle")
# plt.savefig("/home/farscope2/Documents/PhD/Spiking Nets Project/SpikingNetsTexture/graphs/reduced_natural/" +
#            FOLDER_NAME + "Artificial Dataset " + str(xx) + " Texture" + str(yy) + ".pickle" + ".png")
# plt.clf()  # Clear figure post save
plt.show()


# # Create and save heatmap for each tap
# for xx in range(trials):
#     for yy in range(textures):
#         # Open each file individually
#         FILENAME = PATH + "Artificial Dataset " + \
#             str(xx) + "Texture No. " + str(yy) + ".pickle"

#         # Create array of intensities for heatmap
#         with(open(FILENAME, "rb")) as openfile:
#             try:
#                 orig_array = pickle.load(openfile)
#             except EOFError:
#                 print(EOFError)

#         # Remove excess pixels from the top and sides of image
#         for c in range(len(temp_image)):
#             for d in range(len(temp_image[0])):
#                 temp_image[c, d] = orig_array[c+temp_y, d+x_red]  # c+y_red

#         # Remove the excess bottom pixels
#         for y in range(len(reduced_image)):
#             for z in range(len(reduced_image[0])):
#                 reduced_image[y, z] = temp_image[y, z]

#         for u in range(len(reduced_image)):
#             for i in range(len(reduced_image[u])):
#                 intensity[u, i] = len(reduced_image[u, i])

#         # Save dataset seperately for ease of use with network
#         pickle_out = open(os.path.join(data_dir, 'Artificial Dataset ' +
#                                        str(xx) + 'Texture No. ' + str(yy) + '.pickle'), 'wb')
#         pickle.dump(reduced_image, pickle_out, 0)
#         pickle_out.close()

#         # # Plot heatmap of events
#         # plt.imshow(intensity, interpolation='nearest',
#         #            vmin=0, vmax=max_intensity)  # cmap='Reds'
#         # plt.ylabel('Y Pixels')
#         # plt.xlabel('X Pixels')
#         # plt.colorbar()
#         # plt.title("Artificial Dataset Trial " +
#         #           str(xx) + " Texture No. " + str(yy) + " Event Intensity")
#         # plt.savefig("/home/farscope2/Documents/PhD/Spiking Nets Project/SpikingNetsTexture/graphs/reduced_natural/" +
#         #             FOLDER_NAME + "Artificial Dataset " + str(xx) + " Texture" + str(yy) + ".pickle" + ".png")
#         # plt.clf()  # Clear figure post save
#         # # plt.show()

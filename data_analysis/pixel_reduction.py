import numpy as np


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

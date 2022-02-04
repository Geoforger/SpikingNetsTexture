# Import event vision library
import eventvision

# Filename to convert
INPUT_PATH = 'Artificial Dataset 0Texture No. 0.pickle'

# Output filename
OUTPUT_PATH = 'Artificial jAER'

# Convert data and output to j_AER format
in_data = eventvision.read_pickle(INPUT_PATH)
out_data = eventvision.write_j_aer(OUTPUT_PATH)
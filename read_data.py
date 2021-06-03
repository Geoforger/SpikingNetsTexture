import pickle

with(open("SingleTap_run_0_orientation_0.pickle", "rb")) as openfile:
    while True:
        try:
            pickle.load(openfile)
        except EOFError:
            break

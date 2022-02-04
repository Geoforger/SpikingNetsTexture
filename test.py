from pandas import read_hdf, HDFStore, DataFrame,read_csv
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import time, os, json, sys
import numpy as np
from core.utils.data_operations import load_hdf_data, create_hdf
from tempfile import TemporaryFile
from quantities import ms,Hz,s
import elephant, pymuvr
import pickle, itertools
from ttictoc import TicToc
from hyperopt import hp, tpe, fmin
import hyperopt
import ray
from ray.tune import run
from ray.tune.suggest.hyperopt import HyperOptSearch
import ray.tune as tune
from ray.tune.schedulers import AsyncHyperBandScheduler


def main():
    coding_type = 'spatiotemporal'

    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    # Timing
    t = TicToc('name')
    t.tic()

    # Space for parameter search
    space = {
        'tau': hp.quniform('tau', 10, 100, 1),
        'cos': hp.quniform('cos', 0, 1, 0.05)
    }

    best = fmin(objective, space, algo=tpe.suggest, max_evals = 100)  

    t.toc()
    print('Time elapsed: ',t.elapsed)
    print(best)

 
def objective(args):
    tau = args['tau']
    cos = args['cos']
    print('#### Tau ' + str(tau) + ' Cos ' + str(cos)  + ' ####')

	# Data directory
    stimulus_type = 'textures' 
    home_dir = os.path.join(os.environ['DATAPATH'], 'TacTip_NF', 'ABB_' + stimulus_type)
    data_dir_name = '100run-132h-1way-tip3.5'
    data_dir = os.path.join(home_dir, data_dir_name)

	# Load initial taxel positions and metadata
    with open(data_dir + "/meta.json", "r") as read_file:
        meta = json.load(read_file)
    init_taxel_x = [i[0] for i in meta['init_pin_pos']]
    init_taxel_y = [i[1] for i in meta['init_pin_pos']]
    n_pins = len(meta['init_pin_pos'])

	# Target variable
	# Y = np.array(list(range(meta['n_objs']))meta['n_runs'])
    y = []
    for i in range (meta['n_objs']):
        y = np.concatenate((y,np.repeat(i,meta['n_runs'])))

	# X = list(range(11))
	# X = np.array([[x] for x in np.repeat(X,100)])


############################################ ENCODE DATA ################################################################################

    # Spatiotemporal coding	
    if os.path.exists('classificationDataSpatiotemporal.pickle'):
        with open ('classificationDataSpatiotemporal.pickle', 'rb') as fp:
            X = pickle.load(fp)
        # X = X.reshape(-1,1)
    else:
        X = list(np.zeros((meta['n_objs']*meta['n_runs'],1)))
        for obj_idx in range(meta['n_objs']):
            print('Saving data... Object ' + str(obj_idx))
            for run_idx in range(meta['n_runs']):
            # for run_idx in range(1):
                # print('Saving data... Object ' + str(obj_idx) + ' Run ' + str(run_idx))
                data = load_hdf_data(data_dir,obj_idx,run_idx,n_pins)
                frames = data.shape[0]
                spikeTimes = list((data[:,:,5].T)/1000)
                for pin_idx in range(n_pins):
                    spikeTimes[pin_idx] = list(spikeTimes[pin_idx][np.nonzero(spikeTimes[pin_idx])])
                X[obj_idx*meta['n_runs'] + run_idx] = spikeTimes		
        # np.save('classificationDataSpatiotemporal', X)
        with open('classificationDataSpatiotemporal.pickle', 'wb') as f:
            pickle.dump(X, f)

    # with open ('./classificationDataSpatiotemporal.pickle', 'rb') as fp:
    #     X = pickle.load(fp)
        # X = X.reshape(-1,1)
    
    # Flatten data 
    # X = flatten(X)

################################################## PARAMETER OPTIMIZATION ###############################################################

    # Split into training and test sets
    # X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)
    X_train, X_test = [],[]
    y_train, y_test = [],[]
    print('Splitting into training and testing ')

    for obj_idx in range(meta['n_objs']):
        X_train = X_train + X[obj_idx*meta['n_runs']:obj_idx*meta['n_runs']+80]
        X_test = X_test + X[obj_idx*meta['n_runs']+80:(obj_idx+1)*meta['n_runs']]
        y_train = y_train + list(y[obj_idx*meta['n_runs']:obj_idx*meta['n_runs']+80])
        y_test = y_test + list(y[obj_idx*meta['n_runs']+80:(obj_idx+1)*meta['n_runs']])

    # Calculate/load multi-neuron distances
    # if os.path.exists('classificationResultsSpatiotemporal.pickle'):
    # 	with open ('classificationResultsSpatiotemporal.pickle', 'rb') as fp:
    # 		results = pickle.load(fp)
    # else:
    # X_test = X_test[:10]
    print('Calculating distances')
    results = VR_multi_distance(X_test, X_train, tau, cos)
    # with open('classificationResultsSpatiotemporal.pickle', 'wb') as f:
    # 	pickle.dump(results, f)

    # KNN classification
    print('KNN Classification')
    k = 4
    y_pred = [0]*len(y_test)
    for test_run_idx in range(len(X_test)):
        idx = np.argpartition(results[test_run_idx],k)
        neighbours = [int(idx[i])/80 for i in range(k)]
        y_pred[test_run_idx] = np.argmax(np.bincount(neighbours))

    # Record prediction error
    score = (len(y_pred)- np.count_nonzero(np.asarray(y_test)-np.asarray(y_pred)))/len(y_pred)
    result = [cos,tau,score]
    with open('results.pickle', 'ab') as f:
        pickle.dump(result, f)
        
    return 1-score 

#################################################### CLASSIFICATION #####################################################################

	# classification_runs = 10
	# Scores = [0]*classification_runs
	# for classification_run_idx in range(classification_runs):
	
	# 	# Split randomly into training and test sets
	# 	X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)

	# 	# Train and test model 
	# 	if coding_type == 'intensive' or coding_type == 'spatial' or coding_type == 'temporal':
	# 		clf = KNeighborsClassifier(n_neighbors=4, weights='distance', metric='manhattan')
	# 		clf.fit(X_train, y_train) 
	# 		y_pred = clf.predict(X_test)
	# 	else:
	# 		clf = KNeighborsClassifier(n_neighbors=4, weights='distance', metric=VR_distance)
	# 		clf.fit(X_train, y_train)
	# 		y_pred = clf.predict(X_test)

		# # Record prediction error
		# Scores[classification_run_idx] = (len(y_pred)- np.count_nonzero(y_test-y_pred))/len(y_pred)

		# # Print confusion matrix
		# if classification_run_idx ==1: 
		# 	cm = confusion_matrix(y_test,y_pred)
		# 	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		# 	labels = ['Smooth','0.5','1.0','1.5','2.0','2.5','3.0','3.5','4.0','4.5','5.0']
		# 	fig = plt.figure()
		# 	ax = fig.add_subplot(111)
		# 	cax = ax.matshow(cm)
		# 	plt.title(coding_type + ' coding')
		# 	fig.colorbar(cax)
		# 	ax.set_xticks=np.arange(cm.shape[1]),
		# 	ax.set_yticks=np.arange(cm.shape[0]),
		# 	ax.set_xticklabels([''] + labels)
		# 	ax.set_yticklabels([''] + labels)
		# 	plt.xlabel('Predicted')
		# 	plt.ylabel('True')
		# 	plt.savefig('confusion_matrices/confusion_matrix_' + coding_type + '.png')
			# plt.show()

	# print(Scores)
	# print(np.mean(Scores))
	# print(np.std(Scores))

#################################################### CROSS VALIDATION #####################################################################

	# # K-fold Cross validation (leave one out if n_splits = 1100)
	# seed = 2
	# scoring = 'accuracy'
	# kfold = model_selection.KFold(n_splits=10, random_state=seed)
	# # X = np.array([np.array(X[i][0] + [0]*(max_length-len(X[i][0]))) for i in range(len(X))])
	# cv_results = model_selection.cross_val_score(KNeighborsClassifier(), X, y, cv=kfold, scoring=scoring)

	# # Print results
	# msg = "%s: %f (%f)" % ('KNN', cv_results.mean(), cv_results.std())
	# print(msg)
	# t.toc()
	# print('Time elapsed: ',t.elapsed)


######################################### FUNCTIONS ########################################################################################

def VR_distance(spike_trains1_in, spike_trains2_in, tau = 50, cos = 1):
    #ans = elephant.spike_train_dissimilarity.victor_purpura_dist((train1*ms,train2*ms), q=[1.] * Hz, kernel=None, sort=True, algorithm='fast')

	# spike_trains1 = [[list(x[1]) for x in itertools.groupby(spike_trains1_in, lambda x: x==[100000]) if not x[0]]]

	# spike_trains2 = [[list(x[1]) for x in itertools.groupby(spike_trains2_in, lambda x: x==[100000]) if not x[0]]]
	spike_trains1 = spike_trains1_in[np.nonzero(spike_trains1_in)]
	spike_trains2 = spike_trains2_in[np.nonzero(spike_trains2_in)]
	# ans = pymuvr.dissimilarity_matrix(spike_trains1, spike_trains2, cos, tau, 'distance')
	ans = elephant.spike_train_dissimilarity.van_rossum_dist((spike_trains1*ms,spike_trains2*ms), tau=[tau] * ms, sort=True)
	return ans[0][1]

def VR_multi_distance(spike_trains1_in, spike_trains2_in, tau = 50, cos = 1):
    #ans = elephant.spike_train_dissimilarity.victor_purpura_dist((train1*ms,train2*ms), q=[1.] * Hz, kernel=None, sort=True, algorithm='fast')

	# spike_trains1 = [[list(x[1]) for x in itertools.groupby(spike_trains1_in, lambda x: x==[100000]) if not x[0]]]

	# spike_trains2 = [[list(x[1]) for x in itertools.groupby(spike_trains2_in, lambda x: x==[100000]) if not x[0]]]
	spike_trains1 = spike_trains1_in
	spike_trains2 = spike_trains2_in
	ans = pymuvr.dissimilarity_matrix(spike_trains1, spike_trains2, cos, tau, 'distance')
	# ans = elephant.spike_train_dissimilarity.van_rossum_dist((spike_trains1*ms,spike_trains2*ms), tau=[tau] * ms, sort=True)
	return ans

def flatten(X):
	# max_length = max([len(max(X[i],key=len)) for i in range(len(X))])
	max_length = max([sum([len(X[j][i]) for i in range(len(X[j]))]) for j in range(len(X))])+49
	# max_length = 3000
	X_out = [[] for y in range(len(X))] 
	for run_idx in range(len(X)):
		# for pin_idx in range(len(X[run_idx])):
		for pin_idx in range(1):
			X_out[run_idx] =  X_out[run_idx] + [x + 10000*pin_idx for x in X[run_idx][pin_idx]]
	X_out = np.array([np.array(X_out[run_idx] + [0]*(max_length-len(X_out[run_idx]))) for run_idx in range(len(X_out))])
	return X_out

def add_noise(X):
	print(X)

main()
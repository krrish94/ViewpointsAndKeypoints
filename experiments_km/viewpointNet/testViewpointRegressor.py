import math
import numpy as np
import os
import sys

# Set caffe_root so that the right version of PyCaffe is found
caffe_root = '/home/km/libs/caffe-km/'
sys.path.insert(0, caffe_root + 'python')

import cv2
import lmdb

import caffe
from caffe.io import datum_to_array
import caffe.proto.caffe_pb2


if __name__ == '__main__':

	""" Load the Network """
	
	# Set the computation mode to GPU
	caffe.set_mode_gpu()
	# Select GPU with device ID 0 (On this machine, there's only this GPU)
	caffe.set_device(0)

	# Root directory of the codebase
	basedir = '/home/km/code/ViewpointsAndKeypoints/'
	# Directory containing the prototxts
	prototxtDir = basedir + 'prototxts/vggViewpointRegressor/'
	# Directory containing the snapshots
	snapshotDir = basedir + 'snapshots/vggViewpointRegressor/'
	# Directory containing the LMDB files
	lmdbDir = basedir + 'cachedir/VNetTrainFiles/'

	# Load the network, defined in the prototxt file
	net = caffe.Net(prototxtDir + 'deploy.prototxt', snapshotDir + 'net_iter_20000.caffemodel', caffe.TEST)

	# Print layer information
	for key, val in net.blobs.items():
		print key, val.data.shape


	""" Load an image from the LMDB (test) database, and test the network """

	# Open the LMDB (test) database
	lmdb_env = lmdb.open(lmdbDir + 'pascal_imagenet_test_lmdb_data', readonly = True)
	lmdb_txn = lmdb_env.begin()
	lmdb_cursor = lmdb_txn.cursor()
	datum = caffe.proto.caffe_pb2.Datum()

	predictedAzimuths = []
	predictedFeats = []

	i = 0

	for key, val in lmdb_cursor:
		
		# Parse the value to get data into a Caffe 'datum' object
		datum.ParseFromString(val)
		# Read the label (absent if the database contains only an image, in which case label is set to 0)
		label = datum.label
		# Convert the datum object to an nd array
		data = caffe.io.datum_to_array(datum)
		# print data.shape
		
		# Reshape the data (for visualization)

		# data = np.reshape(data, (data.shape[1], data.shape[2], data.shape[0]), order='F')
		
		# The above didn't work (even after trying a lot of variations), so I wrote the below snippet

		# image = np.zeros((data.shape[1], data.shape[2], data.shape[0]))
		# for c in range(data.shape[0]):
		# 	for h in range(data.shape[1]):
		# 		for w in range(data.shape[2]):
		# 			# Note that the channels also flipped. Channel 0 goes to channel 2, etc.
		# 			c2 = 0
		# 			if c == 0:
		# 				c2 = 2
		# 			elif c == 1:
		# 				c2 = 1
		# 			image[h][w][c] = data[c2][h][w]

		# cv2.imshow('test', image)
		# cv2.waitKey(0)

		# Initialize the data layer of the net with the current input
		net.blobs['data'].data[...] = data
		# Run a forward pass and store the output
		out = net.forward()

		# Get the sin and cos of azimuth (and elevation) from the last FC layer
		pred = out['fc8_mod']
		# Compute the actual azimuth
		azimuth = math.atan2(pred[0][0], pred[0][1])*180/math.pi

		predictedAzimuths.append(azimuth)
		predictedFeats.append([pred[0][0], pred[0][1]])

		# Included this break for now, as I want to test only on one image (for debugging the code)
		# break
		i += 1
		if i >= 10:
			break

	# cv2.destroyAllWindows()


	print 'Getting Ground Truth Labels ...'

	# Open the LMDB (test scores) database
	lmdb_env = lmdb.open(lmdbDir + 'pascal_imagenet_test_lmdb_score', readonly = True)
	lmdb_txn = lmdb_env.begin()
	lmdb_cursor = lmdb_txn.cursor()
	datum = caffe.proto.caffe_pb2.Datum()

	trueAzimuths = []
	trueFeats = []

	i = 0

	for key, val in lmdb_cursor:
		
		# Parse the value to get data into a Caffe 'datum' object
		datum.ParseFromString(val)
		# Read the label (absent if the database contains only an image, in which case label is set to 0)
		label = datum.label
		# Convert the datum object to an nd array
		data = caffe.io.datum_to_array(datum)
		# print data.shape

		for d in data:
			angle = math.atan2(d[0][0], d[1][0])*180/math.pi
			trueFeats.append([d[0][0], d[1][0]])

		trueAzimuths.append(angle)

		i += 1
		if i >= 10:
			break

	# Write data to text file
	# outFile = open('net_predictions.txt', 'w')

	testError = 0.0
	for i in range(len(trueAzimuths)):
		# outFile.write(str(testAngles[i]) + ' ' + str(testLabels[i]) + '\n')		
		euclideanLoss = 0;
		for j in range(2):
			euclideanLoss += (predictedFeats[i][j] - trueFeats[i][j])**2
		print math.sqrt(euclideanLoss)	

		# outFile.write(str(testAngles[i]) + ' ' + str(testLabels[i]) + ' ' + str(err) + '\n')


	# print 'Average error: ', testError / len(testLabels), 'over', len(testLabels), 'samples'
	# outFile.write('Average error: ' + str(testError/len(testLabels)) + ' over ' + str(len(testLabels)) + ' samples')

	# outFile.close()

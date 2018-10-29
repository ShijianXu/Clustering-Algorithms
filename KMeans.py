# K-Means Algorithm
# Author: XU Shijian
# 2018.10.27

import numpy as np
import matplotlib.pyplot as plt
import itertools
import copy

class KMeans(object):
	def __init__(self, dataX, k):
		"""
		Normalization can usually improve the result
		Normalize data to [0,1]
		For features of different types, you'd better consider giving weights to different types
		"""
		for i in range(dataX.shape[1]):
			max_ = dataX[:, i].max()
			min_ = dataX[:, i].min()
			dataX[:, i] = (dataX[:, i] - min_) / (max_ - min_)

		self.x = dataX
		self.k = k
		self.num = dataX.shape[0]
		self.pred = np.ones((self.num,))		# the class label of each sample
		print("Shape of x: ", self.x.shape)
		print("shape of prediction: ", self.pred.shape)
		print("Number of samples:", self.num)


	def RanSelCenters(self):
		"""
		randomly select k samples as the initial centers
		"""
		ind = np.random.randint(0, self.num, self.k)
		print("the selected indices: ", ind)
		self.centers = [self.x[i] for i in ind]
		print("Initial centers are :\n", self.centers)


	def compDist(self, index):
		"""
		Compute the distances between the sample x[index] and each center
		"""
		distance = []
		for c in range(self.k):

			"""
			# The commented code is a demo for giveing weights to different types of features
			# In this case, feature is 4-dim, first two and last two are different types
			# Here I give more weights to the first two features when computing distances
			# But usually, normalized feature can directly compute the distance without applying weights

			col_x = self.x[index][:2]
			col_c = self.centers[c][:2]
			loc_x = self.x[index][2:]
			loc_c = self.centers[c][2:]
			dis1 = np.linalg.norm(col_x - col_c) * 40
			dis2 = np.linalg.norm(loc_x - loc_c)
			dist = dis1 + dis2
			"""

			dist = np.linalg.norm (self.x[index]-self.centers[c], ord=2)	# L2 norm

			distance.append(dist)
		distance = np.array(distance)
		return distance


	def updateClasses(self):
		"""
		Update the classes of each sample,
		according to the distance from each center
		"""
		for i in range(self.num):
			distances = self.compDist(i)
			cls_ = np.where(distances==np.min(distances))
			"""
			Sometimes will trigger this error:
			IndexError: index 0 is out of bounds for axis 0 with size 0
			Still don't know why
			"""
			self.pred[i] = cls_[0][0]


	def updateCenters(self):
		"""
		Update the centers
		"""
		changed = False
		for c in range(self.k):
			samples = [self.x[i] for i in range(self.num) if self.pred[i]==c]
			if len(samples)==0:
				# An empty cluster
				# reinitialize this cluster
				print("An empty cluster!")
				ind = np.random.randint(0, self.num, 1)
				self.centers[c] = self.x[ind]
				print("reinitialized with ", self.x[ind])
				changed = True
			else:
				avg = np.mean(samples, axis=0)
				if np.any(self.centers[c] != avg):
					self.centers[c] = avg
					changed = True
		print("Updated centers: ", self.centers)
		return changed


	def fit(self):
		self.RanSelCenters()

		repeat = 100
		for it in range(repeat):
			self.updateClasses()
			changed = self.updateCenters()
			if not changed:
				print("repeat num is: ", it)
				break


	def plotPred(self):
		# This function is used for plot the clustring result for data with 2-dim features
		# Visualization of ground-truth and predicted label
		# Need to predefine enough colors
		colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

		for c in range(self.k):
			class_ = np.array([self.x[i] for i in range(200) if int(self.pred[i])==c])
			plt.scatter(class_[:, 0], class_[:, 1], marker='.', color = colors[c], label=str(c))
		plt.show()


	def evaluate(self):

		"""
		# If you have ground-truth for clustering labels,
		# You can modify the evaluation function by adding a param 'y',
		# i.e., def evaluate(self, y):
		# Then, you can uncomment this code to get the clustring 'accuracy' 

		labels = [j for j in range(self.k)]
		perm = list(itertools.permutations(labels))
		all_acc = []
		for p in perm:
			temp_pred = copy.deepcopy(self.pred)
			for i in range(self.num):
				temp_pred[i] = p[int(self.pred[i])] + 1
			acc = np.sum(temp_pred==y)/self.num
			all_acc.append(acc)

		print("Final accuray is: ", np.max(all_acc))

		# self.plotPred()
		"""

		return self.pred+1	# To make labels start from '1' rather than '0'
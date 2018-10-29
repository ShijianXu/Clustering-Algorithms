# Mean-Shift ALGORITHM
# Author: XU Shijian
# 2018.10.28

import numpy as np
import matplotlib.pyplot as plt
import copy
import itertools
import time

class MeanShift(object):
	"""
	Mean-Shift algorithm with Gaussian kernel
	Update rule:
	x^(t+1) = \sum_i^n( x_i * Norm(x_i | x^(t), h*h*I) ) / \sum_i^n( Norm(x_i | x^(t), h*h*I) )
	"""

	def __init__(self, dataX, h):
		"""
		Normalize data to [0,1]
		"""
		for i in range(dataX.shape[1]):
			max_ = dataX[:, i].max()
			min_ = dataX[:, i].min()
			dataX[:, i] = (dataX[:, i] - min_) / (max_ - min_)

		self.x = dataX
		self.num = dataX.shape[0]
		self.dim = dataX.shape[1]
		self.cov = h**2 * np.eye(self.dim)
		self.pred = np.zeros((self.num, ))

		print("Shape of data: ", self.x.shape)
		self.peak = np.array([np.ones(self.x[0].shape)] * self.num)
		print("Shape of peaks: ", self.peak.shape)


	def compExp(self, i, mu):
		inv = np.linalg.inv(self.cov)
		diff = self.x[i]-mu
		ans = -0.5 * np.matmul(np.matmul(diff, inv), np.transpose(diff))
		return np.exp(ans)


	def compProb(self, i, mu):
		det = np.linalg.det(self.cov)
		coeff = 1 / (np.power(2*np.pi, self.dim/2) * np.power(det, 0.5))
		expo = self.compExp(i, mu)
		return coeff*expo


	def updateSample(self, x_t):
		probs = np.zeros((self.num, ))
		for i in range(self.num):
			probs[i] = self.compProb(i, x_t)

		temp = [self.x[i] * probs[i] for i in range(self.num)]
		res = temp[0]
		for i in range(1, self.num):
			res += temp[i]
		return res / np.sum(probs)


	def converge(self, i):
		start = time.clock()

		x_old = copy.deepcopy(self.x[i])

		# repeat = 300  # Can repeat more times, according to the specific dataset
		repeat = 70

		iterNum = 0
		for it in range(repeat):
			x_new = self.updateSample(x_old)
			iterNum += 1
			self.peak[i] = x_new

			## Relax the condition for convergence
			## =============================================
			# x_new_ = np.round(x_new, 4)
			# x_old_ = np.round(x_old, 4)
			# if np.all(x_new_ == x_old_):
			#	break
			## =============================================
			if np.all(x_new == x_old):
				break


			x_old = copy.deepcopy(x_new)

		end = time.clock()
		print("Sample x_{} converges at {}-th iteration, time: {}".format(i+1, iterNum, str(end-start)))



	def fit(self):
		for i in range(self.num):
			self.converge(i)

		np.save("peaks.npy", self.peak)


	def plotPred(self):
		# This function is used to visualize the clustering result, if the feature of data is 2-dim.
		# Need to predefine enough colors
		colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
		for c in range(self.clusters):
			class_ = np.array([self.x[i] for i in range(self.num) if int(self.pred[i])==c])
			if len(class_)!=0:
				plt.scatter(class_[:, 0], class_[:, 1], marker='.', color = colors[c], label=str(c))
			else:
				print("cluster {} is empty.".format(c))
		plt.show()


	def evaluate(self):
		"""
		Change the round precision according to specific data!
		"""
		self.peak = np.load("peaks_361010.npy")

		print(self.peak)
		peaks = self.peak.tolist()
		unique_peaks = []
		for item in peaks:
			item_round = ['%.4f' % elm for elm in item]
			
			# =================================================
			# item_round_1 = ['%.2f' % elm for elm in item]
			# item_round = ['%.1f' % float(elm) for elm in item_round_1]
			# =================================================

			if item_round not in unique_peaks:
				unique_peaks.append(item_round)
		print(unique_peaks)
		self.clusters = len(unique_peaks)
		print("Number of clusters: ", self.clusters)

		for i in range(self.num):
			pk = self.peak[i]
			pk_ = ['%.4f' % elm for elm in pk]

			# ======================================
			# pk_1 = ['%.2f' % elm for elm in pk]
			# pk_ = ['%.1f' % float(elm) for elm in pk_1]

			ind = unique_peaks.index(pk_)
			self.pred[i] = ind

		# self.plotPred()		# Visualize clustering result, if the feature of data is 2-dim
		return self.pred + 1	# Making the labels star from '1' rather than '0'
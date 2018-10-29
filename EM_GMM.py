# EM Algorithm
# Author: XU Shijian
# 2018.10.27

import numpy as np
import copy
import matplotlib.pyplot as plt
import itertools

class EM_GMM(object):
	def __init__(self, dataX, k):
		# dataX: [N, D], ex. (200,2)
		"""
		Normalize data to [0,1]
		"""
		for i in range(dataX.shape[1]):
			max_ = dataX[:, i].max()
			min_ = dataX[:, i].min()
			dataX[:, i] = (dataX[:, i] - min_) / (max_ - min_)

		self.x = dataX
		self.num = dataX.shape[0]
		self.k = k
		self.dim = dataX.shape[1]
		print("Number of samples: ", self.num)


	def initParams(self):
		self.pi = np.ones((self.k,))/self.k					# initialize pi with equal probs.
		ind = np.random.randint(0, self.num, self.k)		# initialize mu with random samples
		self.mu = [self.x[i] for i in ind]
		self.cov = np.array([np.eye(self.dim)] * self.k)	# initialize cov as diagonal matrix
		print("Initial pi: ", self.pi)
		print("Initial mu: ", self.mu)
		print("Initial cov: ", self.cov)


	def compExp(self, i, j):
		inv = np.linalg.inv(self.cov[j])
		diff = self.x[i]-self.mu[j]
		ans = -0.5 * np.matmul(np.matmul(diff, inv), np.transpose(diff))
		return np.exp(ans)


	def compNormDist(self, i, j):
		det = np.linalg.det(self.cov[j])
		coeff = 1 / (np.power(2*np.pi, self.dim/2) * np.power(det, 0.5))
		expo = self.compExp(i, j)
		return coeff*expo


	def compSumProb(self, i, dists):
		ans = 0
		for j in range(self.k):
			ans += self.pi[j] * dists[i][j]
		return ans


	def E_step(self):
		self.z = np.zeros((self.num, self.k))

		dists = np.zeros((self.num, self.k))
		for i in range(self.num):
			for j in range(self.k):
				dist = self.compNormDist(i, j)
				dists[i][j] = dist
		for i in range(self.num):
			for j in range(self.k):
				self.z[i][j] = (self.pi[j]*dists[i][j]) / self.compSumProb(i, dists)


	def update_mu(self, N):
		for j in range(self.k):
			self.mu[j] = np.zeros(self.mu[j].shape)
			for i in range(self.num):
				self.mu[j] += self.z[i][j] * self.x[i]
			self.mu[j] /= N[j]


	def compDiagVec(self, j, N):
		s = np.zeros((self.dim,))
		for i in range(self.num):
			diff = self.x[i]-self.mu[j]
			s += (diff*diff)*self.z[i][j]
		s = s / N[j]
		return s


	def update_cov(self, N):
		for j in range(self.k):
			self.cov[j] = np.zeros(self.cov[j].shape)
			s_j = self.compDiagVec(j, N)
			for d in range(self.dim):
				self.cov[j][d][d] = s_j[d]


	def M_step(self):
		N = np.array([np.sum(self.z[:, j]) for j in range(self.k)])
		old_pi = copy.deepcopy(self.pi)
		self.pi = N/self.num
		changed = False
		if np.any(self.pi!=old_pi):
			changed = True
		self.update_mu(N)
		self.update_cov(N)

		return changed


	def fit(self):
		self.initParams()
		repeat = 200
		for it in range(repeat):
			self.E_step()
			print("Updated pi: ", self.pi)
			changed = self.M_step()
			
			if not changed:
				print("Repeated {} times.".format(it))
				break


	def plotPred(self, pred):
		# If your the feature of your data is 2-dim,
		# You can use this function to plot the clustring result
		# Need to predefine enough colors
		colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

		for c in range(self.k):
			class_ = np.array([self.x[i] for i in range(self.num) if int(pred[i])==c])
			if len(class_)!=0:
				plt.scatter(class_[:, 0], class_[:, 1], marker='.', color = colors[c], label=str(c))
			else:
				print("cluster {} is empty.".format(c))
		plt.show()


	def evaluate(self):
		"""
		Sometimes, there will be an empty cluster
		"""

		pred = np.zeros((self.num,))
		for i in range(self.num):
			cls_ = np.where(self.z[i, :]==np.max(self.z[i, :]))
			pred[i] = cls_[0][0]

		"""
		# If you have the ground-truth labels 'y' for each point,
		# You can uncomment this code to get the 'accuracy' result,
		# By modifying the function:
		# i.e., def evaluate(self, y):

		labels = [j for j in range(self.k)]
		perm = list(itertools.permutations(labels))
		all_acc = []
		for p in perm:
			temp_pred = copy.deepcopy(pred)
			for i in range(self.num):
				temp_pred[i] = p[int(pred[i])] + 1
			acc = np.sum(temp_pred==y) / self.num
			all_acc.append(acc)

		print("Final accuray is: ", np.max(all_acc))
		# self.plotPred(pred)	# For plot clustering result for data with 2-d features
		"""

		return pred+1	# Making the labels start from '1' rather than '0'
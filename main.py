import numpy as np
from KMeans import KMeans
from EM_GMM import EM_GMM
from MeanShift import MeanShift

def loadData(path):
	data = np.loadtxt(path)
	return np.transpose(data)


def unison_shuffle(x, y):
    p = np.random.permutation(len(x))
    print(p)
    return x[p], y[p]


if __name__ == "__main__":
	dataA_X = "dataset/cluster_data_dataA_X.txt"
	dataA_Y = "dataset/cluster_data_dataA_Y.txt"
	dataB_X = "dataset/cluster_data_dataB_X.txt"
	dataB_Y = "dataset/cluster_data_dataB_Y.txt"
	dataC_X = "dataset/cluster_data_dataC_X.txt"
	dataC_Y = "dataset/cluster_data_dataC_Y.txt"
	
	dataX = loadData(dataC_X)
	dataY = loadData(dataC_Y)

	#dataX, dataY = unison_shuffle(dataX, dataY)

	"""
	kmeans = KMeans(dataX, 4)
	kmeans.fit()
	kmeans.evaluate(dataY)
	"""

	"""
	em = EM_GMM(dataX, 4)
	em.fit()
	em.evaluate(dataY)
	"""

	meanshift = MeanShift(dataX, h = 0.1)
	meanshift.fit()
	meanshift.evaluate()
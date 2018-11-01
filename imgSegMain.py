"""
# This code is implemented by other unkonwn people (Maybe a TA or a Professor from CityU HK)
# Not my implementation!
"""

import numpy as np
from KMeans import KMeans
from EM_GMM import EM_GMM
from MeanShift import MeanShift
import pa2
import pylab as pl
from PIL import Image
import scipy.io as sio


def loadData(path):
	data = np.loadtxt(path)
	return np.transpose(data)


def main():
	import scipy.cluster.vq as vq
	## load and show image
	img = Image.open('images/resized.jpg')

	pl.subplot(1,3,1)
	pl.imshow(img)
    
	## extract features from image (step size = 7)
	X,L = pa2.getfeatures(img, 9)
	

	X_T = np.transpose(X)
	print(X_T)
	
	"""
	Change the methods to one of them: KMeans(X_T, k), EM_GMM(X_T, k), MeanShift(X_T, h)
	"""
	ms = EM_GMM(X_T, k=5)
	ms.fit()
	Y = ms.evaluate()

	"""
	# Sample code, using the methods in scipy.
	# Call kmeans function in scipy.  You need to write this yourself!
	C,Y = vq.kmeans2 (vq.whiten(X.T), 2, iter=1000, minit='random')
	Y = Y + 1 # Use matlab 1-index labeling
	## 
	"""

	# make segmentation image from labels
	segm = pa2.labels2seg(Y,L)
	pl.subplot(1,3,2)
	pl.imshow(segm)
    
	# color the segmentation image
	csegm = pa2.colorsegms(segm, img)
	pl.subplot(1,3,3)
	pl.imshow(csegm)
	pl.show()


if __name__ == "__main__":
	main()
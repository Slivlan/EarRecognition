import cv2, sys
from skimage import feature
import numpy as np

class LBP:
	def __init__(self, num_points=8, radius=2, eps=1e-6, resize=100):
		self.num_points = num_points * radius
		self.radius = radius
		self.eps = eps
		self.resize=resize

	def extract(self, img, numOfBins):
		if (len(img.shape) > 2):
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (self.resize, self.resize))

		lbp = feature.local_binary_pattern(img, self.num_points, self.radius, method="default")
	
		n_bins = int(lbp.max() + 1)
		hist, _ = np.histogram(lbp, density=True, bins=numOfBins, range=(0, n_bins))
		
		# TODO
		
		return hist

if __name__ == '__main__':
	fname = sys.argv[1]
	img = cv2.imread(fname)
	extractor = Extractor()
	features = extractor.extract(img)
	print(features)
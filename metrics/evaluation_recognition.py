import math
import numpy as np

class Evaluation:

	def compute_rank1(self, Y, y):
		#print(f"Y: {Y}")

		#print(f"y: {y}")

		classes = np.unique(sorted(y))
		count_all = 0
		count_correct = 0
		for cla1 in classes:
			idx1 = y==cla1
			if (list(idx1).count(True)) <= 1:
				continue
			# Compute only for cases where there is more than one sample:
			Y1 = Y[idx1==True, :]
			Y1[Y1==0] = math.inf
			for y1 in Y1:
				s = np.argsort(y1)
				smin = s[0]
				imin = idx1[smin]
				count_all += 1
				if imin:
					count_correct += 1
		return count_correct/count_all*100

	def compute_rank1_sum(self, dist, c):
		classes = np.unique(sorted(c))
		count_all = 0
		count_correct = 0
		for cla1 in classes:
			#print(f"c: {c}")
			idx1 = c == cla1
			#print(f"cla1: {cla1}, idx1: {idx1}")
			if (list(idx1).count(True)) <= 1:
				continue
			# Compute only for cases where there is more than one sample:
			Y1 = dist[idx1 == True, :]
			Y1[Y1 == 0] = math.inf #ker delamo test na train setu vrzemo vn tisto identicno sliko
			for y1 in Y1:
				#print("__________")
				avgDistances = []
				for clas in classes:
					idxC = c == clas
					idxInf = y1!=math.inf
					idxSkupaj = np.logical_and(idxC, idxInf)
					y1C = y1[idxSkupaj == True]
					#print(y1C)
					sumOfC = np.average(y1C)
					#print(f"For picture of class {cla1}, the average of distances for class {clas} is: {sumOfC}")
					avgDistances.append(sumOfC)

				s = np.argsort(avgDistances)
				smin = s[0]
				imin = classes[smin]
				#print(f"min class for actual class {cla1}: {imin}")
				count_all += 1
				if imin == cla1:
					#print(f"min class for actual class {cla1}: {imin}")
					#print("success!")
					count_correct += 1
		return count_correct / count_all * 100


	def compute_rank1_avg_top3(self, dist, c):
		classes = np.unique(sorted(c))
		count_all = 0
		count_correct = 0
		for cla1 in classes:
			#print(f"c: {c}")
			idx1 = c == cla1
			#print(f"cla1: {cla1}, idx1: {idx1}")
			if (list(idx1).count(True)) <= 1:
				continue
			# Compute only for cases where there is more than one sample:
			Y1 = dist[idx1 == True, :]
			Y1[Y1 == 0] = math.inf #ker delamo test na train setu vrzemo vn tisto identicno sliko
			for y1 in Y1:
				#print("__________")
				avgDistances = []
				for clas in classes:
					idxC = c == clas
					idxInf = y1!=math.inf
					idxSkupaj = np.logical_and(idxC, idxInf)
					y1C = y1[idxSkupaj == True]
					#print(y1C)
					y1C = np.sort(y1C)
					sumOfC = np.average(y1C[:3])
					#print(f"For picture of class {cla1}, the average of distances for class {clas} is: {sumOfC}")
					avgDistances.append(sumOfC)

				s = np.argsort(avgDistances)
				smin = s[0]
				imin = classes[smin]
				#print(f"min class for actual class {cla1}: {imin}")
				count_all += 1
				if imin == cla1:
					#print(f"min class for actual class {cla1}: {imin}")
					#print("success!")
					count_correct += 1
		return count_correct / count_all * 100



	# Add your own metrics here, such as rank5, (all ranks), CMC plot, ROC, ...

		# def compute_rank5(self, Y, y):
	# 	# First loop over classes in order to select the closest for each class.
	# 	classes = np.unique(sorted(y))
		
	# 	sentinel = 0
	# 	for cla1 in classes:
	# 		idx1 = y==cla1
	# 		if (list(idx1).count(True)) <= 1:
	# 			continue
	# 		Y1 = Y[idx1==True, :]

	# 		for cla2 in classes:
	# 			# Select the closest that is higher than zero:
	# 			idx2 = y==cla2
	# 			if (list(idx2).count(True)) <= 1:
	# 				continue
	# 			Y2 = Y1[:, idx1==True]
	# 			Y2[Y2==0] = math.inf
	# 			min_val = np.min(np.array(Y2))
	# 			# ...
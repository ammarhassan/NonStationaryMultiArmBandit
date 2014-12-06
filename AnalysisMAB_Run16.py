"""
This file parse the results of the algorithms; load them in appropriate datastructures; plots and do statistics
Run15
"""
import numpy as np
import os
from conf import *
from matplotlib.pylab import *
from operator import itemgetter

# store article stats
class article():
	def __init__(self):
		self.tim = []
		self.theta = []
		self.Oucba = []
		self.Oucbc = []
		self.OucbCTR = []
		self.ucbc = []
		self.ucba = []
		self.ucbCTR = []
		self.Ogreedyc = []
		self.Ogreedya = []
		self.OgreedyCTR = []
		self.Orandc = []
		self.Oranda = []
		self.OrandCTR = []
		self.resetCount = []
		self.varClus = []
		self.means = []

		self.inPool = []
		self.selected = []

		self.resetSessionUCBCtr = []

	def updateresetSessionUCBCtr(self):
		self.resetSessionUCBCtr.append(self.ucbCTR[-1])

	def update(self, tim, ucbAa, ucbAc, Oranda, Orandc, Ogreedya, Ogreedyc, ucba, ucbc, theta, resetCount, varClus, means):
		self.tim.append(tim)
		self.ucbc.append(ucbAc)
		self.ucba.append(ucbAa)
		self.ucbCTR.append(ucbAc/ucbAa)
		self.Orandc.append(Orandc)
		self.Oranda.append(Oranda)
		self.OrandCTR.append(Orandc/Oranda)
		self.Ogreedyc.append(Ogreedyc)
		self.Ogreedya.append(Ogreedya)
		self.OgreedyCTR.append(Ogreedyc/Ogreedya)
		self.Oucba.append(ucba)
		self.Oucbc.append(ucbc)
		self.OucbCTR.append(ucbc/ucba)
		self.theta.append(theta)
		self.resetCount.append(resetCount)
		self.varClus.append(varClus)
		self.means.append(means)

	def updatePoolCount(self, count):
		self.inPool.append(count)

	def updateSelected(self, count):
		self.selected.append(count)

	# for ease of manipulation the data is converted into numpy arrays
	def done(self):
		for key in self.__dict__.keys():
			self.__dict__[key] = np.array(self.__dict__[key])

def fill_DS(filename, articles, resetCount):
	print filename
	with open(filename, 'r') as f:
		# for i in range(3):
			# line = f.readline()
			# print line
		for line in f:
			words = line.split(',')
			if words[0].strip() != "data":
				continue
			tim = int(words[1])

			ucba, ucbc, randa, randc, greedya, greedyc = [float(x) for x in words[2].split(';')]
			# add preferences for each article
			if words[3].strip() and words[4].strip() and words[5].strip():
				for x in words[3].split(';'):
					varClus, ucbAa, ucbAc, ids, theta = x.split(' ') #[float(y) for y in x.split(' ')[:3]]
					ucbAa, ucbAc, ids = float(ucbAa), float(ucbAc), float(ids)
					varClus = [float(x) for x in varClus.split('|')]
					theta = [float(x) for x in theta.split('|')]
					# means = produce_means(theta, 5)
					means = []
					# theta = [float(y) for y in x.split(' ')[3:]]
					if ids not in articles:
						articles[ids] = article()
					articles[ids].update(tim, ucbAa, ucbAc, randa, randc, greedya, greedyc, ucba, ucbc, theta, resetCount, varClus, means)

			# add number of times the article was in the pool
				for x in words[4].strip().split(';'):
					ids, count = [float(y) for y in x.split(' ')]
					if ids not in articles:
						articles[ids] = article()
					articles[ids].updatePoolCount(count)

			# add the number of times the article was selected from the pool
				for x in words[5].strip().split(';'):
					ids, count = [float(y) for y in x.split(' ')]
					if ids not in articles:
						articles[ids] = article()
					articles[ids].updateSelected(count)
		# except:
		# 	print line
	return articles

def summary(articles, variableName):
	return [(x, articles[x].__dict__[variableName][-1]) for x in articles]

def produce_means(theta, num_clusters):
	means = np.zeros(num_clusters)

	for i in range(num_clusters):
		featureVector = np.zeros(num_clusters)
		featureVector[i] = 1
		means[i] = np.dot(theta, featureVector)
	return means

if __name__ == '__main__':
	reloads = 0
	if reloads:
		filenames =[x for x in os.listdir(save_address) if '.csv' in x]
		# dictionaries for three modes of data
		articlesSingle = {}
		articlesMultiple = {}
		articlesHours = {}

		# calculating how many times theta was reset. Theta stays static in a file.
		countSingle = 1
		countMultiple = 1
		countHours = 1
		for x in filenames:
			if 'single' in x:
				articlesSingle = fill_DS(os.path.join(save_address, x), articlesSingle, countSingle)
				countSingle = countSingle + 1
			elif 'multiple' in x:
				articlesMultiple = fill_DS(os.path.join(save_address,x), articlesMultiple, countMultiple)
			elif '4hours' in x:
				articlesHours = fill_DS(os.path.join(save_address,x), articlesHours, countHours)
				countHours = countHours + 1

		# converting to Numpy arrays
		for x in articlesSingle:
			articlesSingle[x].done()

		for x in articlesMultiple:
			articlesMultiple[x].done()

		for x in articlesHours:
			articlesHours[x].done()


	# finding the difference of CTR after and before 
	ass = summary(articlesHours, 'ucbCTR')
	asm = summary(articlesMultiple, 'ucbCTR')
	diff = [(str(x[0][0]), str(x[1][0]), x[0][1] - x[1][1]) for x in zip(ass, asm)]
	diff = sorted(diff, key = itemgetter(2))
	differences = [x[2] for x in diff]
	print '\n'.join([str(x[0]) + ', ' + str(x[1]) + ', ' + str(x[2]) for x in diff])

	id = 109670
	objS = articlesSingle[id]
	objH = articlesHours[id]
	objC = objH
	objM = articlesMultiple[id]

	# tim = np.array(range(shape(articlesSingle[id][0])[0]))
	maxClick = objC.ucbc[-1]
	# calculating batch stats
	with np.errstate(invalid='ignore'):
		accessSingleBatches = np.concatenate((np.array([objC.ucba[0]]), objC.ucba[1:] - objC.ucba[:-1]))
		clickSingleBatches = np.concatenate((np.array([objC.ucbc[0]]), objC.ucbc[1:] - objC.ucbc[:-1]))
		batchSingleCTR = clickSingleBatches / accessSingleBatches

		accessMultBatches = np.concatenate((np.array([objM.ucba[0]]), objM.ucba[1:] - objM.ucba[:-1]))
		clickMultBatches = np.concatenate((np.array([objM.ucbc[0]]), objM.ucbc[1:] - objM.ucbc[:-1]))
		batchMultCTR = clickMultBatches / accessMultBatches

	yAccMax =max([max(accessMultBatches), max(accessSingleBatches)])
	yCTRMax =max([max(batchMultCTR), max(batchSingleCTR)])
	thetaMax = np.max(objC.theta)
	thetaMin = np.min(objC.theta)
	minTim, maxTim = min(objC.tim), max(objC.tim)
	if 1:
		f, axarr = plt.subplots(4, sharex=True)
		axarr[0].set_ylim([thetaMin, thetaMax])
		axarr[0].plot(objC.tim, objC.theta)
		axarr[0].set_ylabel('Daily \nPreferences')
		axarr[0].set_title('title ID : ' + str(id) + 'Sessions:' + ','.join([str(x) for x in list(set(objH.resetCount))]) + ' Days:' + ','.join([str(x) for x in list(set(objS.resetCount))]))

		axarr[1].set_ylim([thetaMin, thetaMax])
		axarr[1].plot(objM.tim, objM.theta)
		axarr[1].set_ylabel('Static \nPreferences')

		# axarr[0].set_title('Plots of Daily Dynamic Theta for Min Diff CTR')
		axarr[2].set_ylim([0, yAccMax])
		axarr[2].plot(objC.tim, accessSingleBatches)
		axarr[2].plot(objM.tim, accessMultBatches, 'm')
		axarr[2].set_ylabel('Dynamic \nUCB Access', color='b')
		axarrT = axarr[2].twinx()
		axarrT.plot(objC.tim, objC.ucbCTR, 'r')
		axarrT.plot(objM.tim, objM.ucbCTR, 'g')
		axarrT.plot(objM.tim, objM.OrandCTR, 'y')
		axarrT.set_ylabel('Dynamic CTR', color='r')
		# axarr[3].plot(objC.tim, objC.inPool, '.')
		# axarr[3].plot(objC.tim, objC.selected, '.')
		# axarr[3].plot(objM.tim, objM.inPool, '.')
		# axarr[3].plot(objM.tim, objM.selected, '.')
		# axarr[3].legend(['In Pool', 'Selected'])
		axarr[3].plot(objC.tim, objC.varClus)
		# axarr[4].plot(objS.tim, objM.means)

		xlabel('days:' + ','.join([str(x) for x in list(set(objC.resetCount))]))

	# from reset_count find the articles


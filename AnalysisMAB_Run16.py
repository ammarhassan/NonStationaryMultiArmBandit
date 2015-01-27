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
		self.restartTimes = []

		self.inPool = []
		self.selected = []

		self.resetSessionUCBCtr = []

	def updateRestartTimes(self, tim):
		self.restartTimes.append(tim)

	def updateresetSessionUCBCtr(self):
		self.resetSessionUCBCtr.append(self.ucbCTR[-1])

	def update(self, tim, ucbAa, ucbAc, Oranda, Orandc, Ogreedya, Ogreedyc, ucba, ucbc, theta, resetCount, varClus, means):
		self.tim.append(tim)
		self.ucbc.append(ucbAc)
		self.ucba.append(ucbAa)
		if ucbAa > 0:
			self.ucbCTR.append(ucbAc/ucbAa)
		else:
			self.ucbCTR.append(0)
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
		self.varClus= np.sum(self.varClus, axis=1)

def fill_DS(filename, articles, resetCount):
	print filename
	firstTime=True
	with open(filename, 'r') as f:
		# for i in range(3):
			# line = f.readline()
			# print line
		for line in f:
			words = line.split(',')
			if words[0].strip() != "data":
				continue
			tim = int(words[1])
			if firstTime:
				firstTime=False
				for x in articles:
					articles[x].updateRestartTimes(tim)
				

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
	performanceMult = []
	performanceHr = []
	parameter = []
	stats = 0
	reloads = 0
	plots_ = True
	MultipleAnalysis = True
	print 'reloads', reloads, 'stats', stats
	for alp in range(1,2):
		alpha = .6
		decay = 1 - 1.0/(4**13)

		# alpha = alp * .1
		if reloads:
			# filenames =[x for x in os.listdir(save_address) if '.csv' in x and 'alpha='+str(alpha) in x]
			filenames =[x for x in os.listdir(save_address) if '.csv' in x and 'Decay='+str(decay) in x and 'alpha='+str(alpha) in x]
			# filenames =[x for x in os.listdir(save_address) if '.csv' in x]
			if filenames:
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
						# print articlesMultiple[109473].OucbCTR[-1]
					elif 'hours4' in x or '4hours' in x:
						articlesHours = fill_DS(os.path.join(save_address,x), articlesHours, countHours)
						countHours = countHours + 1

				# converting to Numpy arrays
				for x in articlesSingle:
					articlesSingle[x].done()

				for x in articlesMultiple:
					articlesMultiple[x].done()

				for x in articlesHours:
					articlesHours[x].done()

			# performanceMult.append(articlesMultiple[109530].OucbCTR[-1] / articlesMultiple[109530].OrandCTR[-1])
			# parameter.append(alp)
			# performanceHr.append(articlesHours[109530].OucbCTR[-1])

			# finding the difference of CTR after and before 
		if stats:
			ass = summary(articlesHours, 'ucbCTR')
			asm = summary(articlesMultiple, 'ucbCTR')
			diff = [(str(x[0][0]), str(x[1][0]), x[0][1] - x[1][1]) for x in zip(ass, asm)]
			diff = sorted(diff, key = itemgetter(2))
			differences = [x[2] for x in diff]
			print '\n'.join([str(x[0]) + ', ' + str(x[1]) + ', ' + str(x[2]) for x in diff])
		if plots_ and not MultipleAnalysis:
			id = 109586
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
				batchChosenCTR = clickSingleBatches / accessSingleBatches

				accessMultBatches = np.concatenate((np.array([objM.ucba[0]]), objM.ucba[1:] - objM.ucba[:-1]))
				clickMultBatches = np.concatenate((np.array([objM.ucbc[0]]), objM.ucbc[1:] - objM.ucbc[:-1]))
				batchMultCTR = clickMultBatches / accessMultBatches

				accessMultrandBatches = np.concatenate((np.array([objM.Oranda[0]]), objM.Oranda[1:] - objM.Oranda[:-1]))
				clickMultrandBatches = np.concatenate((np.array([objM.Orandc[0]]), objM.Orandc[1:] - objM.Orandc[:-1]))
				batchMultrandCTR = clickMultrandBatches / accessMultrandBatches

			yAccMax =max([max(accessMultBatches), max(accessSingleBatches)])
			yCTRMax =max([max(batchMultCTR), max(batchChosenCTR)])
			thetaMax = np.max(objC.theta)
			thetaMin = np.min(objC.theta)
			minTim, maxTim = min(objC.tim), max(objC.tim)
			if 1:
				f, axarr = plt.subplots(4, sharex=True)
				axarr[0].set_ylim([thetaMin, thetaMax])
				axarr[0].plot(objC.tim, objC.theta)

				axarr[0].set_ylabel('Daily \nPreferences')
				axarr[0].set_title("Fluctuating popularity of article")
				# axarr[0].set_title('title ID : ' + str(id) + 'Sessions:' + ','.join([str(x) for x in list(set(objH.resetCount))]) + ' Days:' + ','.join([str(x) for x in list(set(objS.resetCount))]))

				axarr[1].set_ylim([thetaMin, thetaMax])
				axarr[1].plot(objM.tim, objM.theta)
				axarr[1].set_ylabel('Static \nPreferences')

				# axarr[0].set_title('Plots of Daily Dynamic Theta for Min Diff CTR')
				axarr[2].set_ylim([0, yAccMax])
				axarr[2].plot(objC.tim, accessSingleBatches)
				# axarr[2].plot(objC.tim, objS.ucba)
				axarr[2].plot(objM.tim, accessMultBatches, 'm')
				# axarr[2].plot(objM.tim, objM.ucba, 'm')

				axarr[2].set_ylabel('Dynamic \nUCB Access', color='b')
				axarrT = axarr[2].twinx()
				axarrT.plot(objC.tim, batchChosenCTR, 'r')
				axarrT.plot(objM.tim, batchMultCTR, 'g')
				axarrT.plot(objM.tim, batchMultrandCTR, 'y')
				axarrT.set_ylabel('Dynamic CTR', color='r')
				# axarr[3].plot(objC.tim, objC.inPool, '.')
				# axarr[3].plot(objC.tim, objC.selected, '.')
				# axarr[3].plot(objM.tim, objM.inPool, '.')
				# axarr[3].plot(objM.tim, objM.selected, '.')
				# axarr[3].legend(['In Pool', 'Selected'])
				axarr[3].plot(objC.tim, objC.varClus*alpha, objM.tim, objM.varClus*alpha)
				# axarr[4].plot(objS.tim, objM.means)

				axarr[2].set_xlabel('title ID : ' + str(id) + ' Sessions:' + ','.join([str(x) for x in list(set(objH.resetCount))]) + ' Days:' + ','.join([str(x) for x in list(set(objS.resetCount))]))
				# xlabel('days:' + ','.join([str(x) for x in list(set(objC.resetCount))]))
				for t in articlesHours[id].restartTimes:
					if t > minTim and t < maxTim:
						xSet = [t for it in range(31)]
						maxCTR = max([max(batchChosenCTR), max(batchMultCTR), max(batchMultrandCTR)])
						ySet = (np.array(range(0, 31))*1.0/30)*maxCTR
						# axarr[0].plot(xSet, ySet, 'black')
						# axarr[1].plot(xSet, ySet, 'black')
						axarrT.plot(xSet, ySet, 'black')
			# from reset_count find the articles

		if plots_ and MultipleAnalysis:
			id = 109584
			objM = articlesMultiple[id]

			maxClick = objM.ucbc[-1]
			# calculating batch stats
			with np.errstate(invalid='ignore'):
				accessMultBatches = np.concatenate((np.array([objM.ucba[0]]), objM.ucba[1:] - objM.ucba[:-1]))
				clickMultBatches = np.concatenate((np.array([objM.ucbc[0]]), objM.ucbc[1:] - objM.ucbc[:-1]))
				batchMultCTR = clickMultBatches / accessMultBatches

				accessMultrandBatches = np.concatenate((np.array([objM.Oranda[0]]), objM.Oranda[1:] - objM.Oranda[:-1]))
				clickMultrandBatches = np.concatenate((np.array([objM.Orandc[0]]), objM.Orandc[1:] - objM.Orandc[:-1]))
				batchMultrandCTR = clickMultrandBatches / accessMultrandBatches

			yAccMax =max(accessMultBatches)
			yCTRMax =max(batchMultCTR)
			thetaMax = np.max(objM.theta)
			thetaMin = np.min(objM.theta)
			minTim, maxTim = min(objM.tim), max(objM.tim)

			f, axarr = plt.subplots(3, sharex=True)
			axarr[0].set_ylim([thetaMin, thetaMax])
			axarr[0].plot(objM.tim, objM.theta)
			axarr[0].set_ylabel('Decaying \nParameters')

			axarr[1].set_ylim([0, yAccMax])
			axarr[1].plot(objM.tim, accessMultBatches, 'm')
			axarr[1].set_ylabel('Dynamic \nUCB Access', color='b')

			axarrT = axarr[1].twinx()
			axarrT.plot(objM.tim, batchMultCTR, 'g')
			axarrT.plot(objM.tim, batchMultrandCTR, 'y')
			axarrT.set_ylabel('Dynamic CTR', color='r')

			# axarr[1].set_xlabel('title ID : ' + str(id) + ' Sessions:' + ','.join([str(x) for x in list(set(objH.resetCount))]) + ' Days:' + ','.join([str(x) for x in list(set(objS.resetCount))]))

			axarr[2].plot(objM.tim, objM.varClus*alpha)
			

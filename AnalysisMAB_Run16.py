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
class overallStats():
	def __init__(self):
		self.tim = []
		self.Oucba = []
		self.Oucbc = []
		self.OucbCTR = []
		self.Ogreedyc = []
		self.Ogreedya = []
		self.OgreedyCTR = []
		self.Orandc = []
		self.Oranda = []
		self.OrandCTR = []
		self.restartTimes = []
		self.inPool = []
		self.selected = []

	def update(self,  tim, Oranda, Orandc, Ogreedya, Ogreedyc, Oucba, Oucbc):
		self.tim.append(tim)
		self.Orandc.append(Orandc)
		self.Oranda.append(Oranda)
		self.OrandCTR.append(Orandc/Oranda)
		self.Ogreedyc.append(Ogreedyc)
		self.Ogreedya.append(Ogreedya)
		self.OgreedyCTR.append(Ogreedyc/Ogreedya)
		self.Oucba.append(Oucba)
		self.Oucbc.append(Oucbc)
		self.OucbCTR.append(Oucbc/Oucba)

	def updateRestartTimes(self, tim):
		self.restartTimes.append(tim)

	def done(self):
		for key in self.__dict__.keys():
			self.__dict__[key] = np.array(self.__dict__[key])

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


def fill_DS(filename, articles, resetCount, Ostats):
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
				Ostats.updateRestartTimes(tim)
				

			ucba, ucbc, randa, randc, greedya, greedyc = [float(x) for x in words[2].split(';')]
			Ostats.update(tim, randa, randc, greedya, greedyc, ucba, ucbc)
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
	return articles, Ostats

def summary(articles, variableName):
	return [(x, articles[x].__dict__[variableName][-1]) for x in articles]

def loadExperiment(indicators, save_address, Ostats):
	def indicatorsInFile(filename):
		for indicator in indicators:
			if indicator not in filename:
				return False
		return True
	filenames = [x for x in os.listdir(save_address) if indicatorsInFile(x)]
	articlesDict = {}
	for fileCount, x in enumerate(filenames):
		articlesDict, Ostats = fill_DS(os.path.join(save_address, x),
		 articlesDict, fileCount, Ostats)
	for key in articlesDict:
		articlesDict[key].done()
	Ostats.done()
	return articlesDict, Ostats


def produce_means(theta, num_clusters):
	means = np.zeros(num_clusters)

	for i in range(num_clusters):
		featureVector = np.zeros(num_clusters)
		featureVector[i] = 1
		means[i] = np.dot(theta, featureVector)
	return means

def intervalCalculation(values, bins):
	binsIndex = []
	for v in values:
		for ind, g in enumerate(bins[1:]):
			if v < g:
				binsIndex.append(ind)
				break
	return binsIndex

def calBinStats(function, intervals_values, numBins):
	binValues = [[] for i in range(numBins)]
	binStats = np.zeros(numBins)
	# gather the data of each bin in binValues[bin_number]
	for interval, value in intervals_values:
		binValues[interval].append(value)
	# apply the function on the bin data
	for ind, x in enumerate(binValues):
		binStats[ind] = function(x)
	# for each element in the original array copy the bin statistic
	# interval_stats = np.zeros(len(intervals_values))
	# for ind, data in enumerate(intervals_values):
	# 	interval, value = data
	# 	interval_stats[ind] = binStats[interval]
	return binStats


def plotDiffParameter(articlesLinUCB, articlesDecayLinUCB):
	keys_ = set(articlesLinUCB.keys()).intersection(articlesDecayLinUCB.keys())
	diff = [(np.linalg.norm(articlesLinUCB[x].theta[-1] - articlesDecayLinUCB[x].theta[-1])) for x in keys_]

	n, bins, patches = hist(diff)
	intervals_index = intervalCalculation(diff, bins)

	decayUCBClicks = calBinStats(sum, zip(
		intervals_index, 
		zip(*summary(articlesDecayLinUCB, "ucbc"))[1]
		), len(bins) - 1
	)
	decayUCBClicks = (1.0 * decayUCBClicks / sum(decayUCBClicks)) * 100
	plot((bins[1:]+bins[:-1])/2, decayUCBClicks, "r")
	return 1

def evaluateAlgorithmTest(parameterInd, parameterFunc, parameterName, labels, directory):
	parameters, parameterCTR = [], []
	for alp in parameterInd:
		param = parameterFunc(alp)
		OStats = overallStats()
		articlesDecayLinUCB, OStats = loadExperiment(
			[parameterName+'='+str(param)]+labels,
			os.path.join(save_address,directory),
			OStats,
			)
		parameters.append(alp)
		parameterCTR.append(OStats.OucbCTR[-1])

	# 	plot(OStats.tim, OStats.OucbCTR)
	# legend([str(parameterFunc(i)) for i in parameterInd], loc=4)
	print parameterCTR
	# plot(parameters, parameterCTR)
	# xlabel("alpha index; alpha = .1*index")
	# # xlabel("time in seconds")
	# ylabel("Absolute CTR")
	# title("Training performance of LinUCB for different alpha Parameters")

def plotArticleCurves(articles, articlesIds, OStats, differences):
	def calculateY(top, bottom, ctr):
		last = bottom + (top - bottom) * ctr
		return last

	minTim = min(OStats.tim)
	maxTim = max(OStats.tim)

	minCTR = min(map(lambda x: min(articles[x].ucbCTR), articles))
	maxCTR = max(map(lambda x: max(articles[x].ucbCTR), articles))
	
	# minCTR = min(map(lambda x: x[1], differences))
	# maxCTR = max(map(lambda x: x[1], differences))	

	# ylength = (maxCTR - minCTR)

	# plot()
	# xlim([minTim-5000, maxTim+5000])
	# ylim([minCTR, maxCTR])
	# y = 0
	# diff = maxCTR - minCTR
	f, axarr = plt.subplots(len(differences), sharex=True)
	axarr[0].set_xlim([minTim, maxTim])
	# for x, diffs in differences:
		# y = calculateY( maxCTR, minCTR, diffs)
		# ucba = np.concatenate((np.array([articles[x].ucba[0]]), articles[x].ucba[1:] - articles[x].ucba[:-1]))
		# ucbc = np.concatenate((np.array([articles[x].ucbc[0]]), articles[x].ucbc[1:] - articles[x].ucbc[:-1]))
		# ucbCTR = ucbc / ucba
		# plot(articles[x].tim, articles[x].ucbCTR + y)
	for index, item in enumerate(differences):
		key, diffs = item
		axarr[index].set_ylim([minCTR, maxCTR])
		ucba = np.concatenate((np.array([articles[key].ucba[0]]), articles[key].ucba[1:] - articles[key].ucba[:-1]))
		ucbc = np.concatenate((np.array([articles[key].ucbc[0]]), articles[key].ucbc[1:] - articles[key].ucbc[:-1]))
		ucbCTR = ucbc / ucba
		axarr[index].plot(articles[key].tim, ucbCTR)
		for t in OStats.restartTimes:
			xSet = [t for it in range(31)]
			
			ySet = minCTR + (np.array(range(31))*1.0/30)*(maxCTR - minCTR)
			axarr[index].plot(xSet, ySet, 'black')

def plotLines(axes_, xlocs):
	for xloc, color in xlocs:
		# axes = plt.gca()
		# print xloc
		for x in xloc:
			xSet = [x for _ in range(31)]
			ymin, ymax = axes_.get_ylim()
			ySet = ymin + (np.array(range(0, 31))*1.0/30) * (ymax - ymin)
			axes_.plot(xSet, ySet, color)
			# print xSet[0], xSet[-1], ySet[0], ySet[-1]

def plotHorizontalLines(axes_, locations):
	for sloc, eloc in locations:
		xSet = sloc + np.array(range(0,31))/30.0 * (eloc - sloc)
		ySet = np.ones(31) * random()
		axes_.plot(xSet, ySet)

def getBatchStats(arr):
	return np.concatenate((np.array([arr[0]]), np.diff(arr)))

def drawAndSavePlots(xAx, ucbCtr, decucbCtr, pointsPerPlots, lineLocs, articleLines, name):
	for i in range(xAx[-1]//pointsPerPlots+1):
		f, axarr = plt.subplots(1, sharex=True)
		st, fin = i*pointsPerPlots,(i+1)*pointsPerPlots
		fin = fin if fin < len(xAx) else len(xAx)
		axarr.plot(xAx[st:fin], ucbCtr[st:fin], 'r', xAx[st:fin], decucbCtr[st:fin], 'b')
		plotLines(axarr, [([x for x in lineLocs if x <= fin and x>=st], 'black')])
		# plotHorizontalLines(axarr[1], articleLines)
		# plotLines(axarr[1], [([x for x in lineLocs if x <= fin and x>=st], 'black')])
		legend(["LinUCB", "DecLinUCB"])
		# ylabel("Batch CTR")
		# xlabel("Time stamp index")
		# savefig(save_address+"fig_%s.pdf"%i, format="pdf")
		f.savefig(name)
		return f

def aggregate(tim, array, bin=None):
	if bin:
		return [sum(array[bin*x:bin*(x+1)]) for x in range(len(tim)//bin + 1)]
	else:
		aggArray = dict([(i,0) for i in list(set(tim))])
		for t, a in zip(tim, array):
			aggArray[t] += a
		aggArray = sorted(aggArray.items(), key=itemgetter(0))
		return np.array(zip(*aggArray)[1])


def tim_index(timeArray, tim, bin=None):
	if bin:
		return [i for i, x in enumerate(timeArray) if x==tim][0]//bin
	else:
		aggArray = sorted(list(set(timeArray)))
		for t, x in enumerate(aggArray):
			if x==tim:
				return t
		return None

if __name__ == '__main__':

	stats = 0
	reloads = 1
	plots_ =0
	save_special = 1
	bin = 100
	analysistype = "decayLinUCB"
	print 'reloads', reloads, 'stats', stats

	for day in ['03']:#, '04', '05', '06', '07', '08', '09', '10']:

		if reloads:
			ucbStats = overallStats()
			decucbStats = overallStats()

			parameter = []
			parameterCTR = []

			# evaluateAlgorithmTest(range(1,10),lambda x: x , "Decay",
			# ['.csv', 'multiple', 'alpha=3'], "Run24")

			# the evaluation of LinUCB with changing alpha parameter
			# evaluateAlgorithmTest(range(1,10),lambda x: x*.1 , "alpha",
			# 	['.csv', 'multiple'], "Run19")

			alpha = 3
			decay = 1 - 1.0/(4**5)
			# decay = 7

			articlesLinUCB, ucbStats = loadExperiment(
				['.csv', 'multiple', 'alpha='+str(alpha), day+"_03_28"], 
				os.path.join(save_address,'Run27'),
				ucbStats,
				)
			articlesDecayLinUCB, decucbStats = loadExperiment(
				['.csv', 'multiple', 'Decay=8', 'alpha=3', day+'_03_24'],
				os.path.join(save_address,'Run27'),
				decucbStats,
				)

	
		startTimes = [tim_index(ucbStats.tim, y.tim[0], bin) for x,y in articlesLinUCB.items()]
		endTimes = [tim_index(ucbStats.tim, y.tim[-1], bin) for x,y in articlesLinUCB.items()]
		locations = zip(startTimes, endTimes)
		decucbCtr_b = getBatchStats(aggregate(decucbStats.tim, decucbStats.Oucbc, bin)) / getBatchStats(aggregate(decucbStats.tim, decucbStats.Oucba, bin))
		ucbCtr_b = getBatchStats(aggregate(ucbStats.tim, ucbStats.Oucbc, bin)) / getBatchStats(aggregate(ucbStats.tim, ucbStats.Oucba, bin))

		# tempDic = []
		# for x,y in articlesLinUCB.items():
		# 	stime = tim_index(ucbStats.tim, y.tim[0], bin)
		# 	etime = [x for x in startTimes if x > stime][0]
		# 	ctrArray = getBatchStats(aggregate(decucbStats.tim, decucbStats.Oucbc, bin)) / getBatchStats(aggregate(decucbStats.tim, decucbStats.Oucba, bin))
		# 	for t in range(stime, etime):
		# 		tempDic[t].append()

		binLength = str(500*bin)
		name = "/Users/Ammar/GradSchool/MastersThesis/implementation/Yahoo/NonStationaryMultiArmBandit/linUCBvsDecLinUCB_day"+day+"_batchSize"+binLength+"decay=1-1e-8.pdf"
		fig = drawAndSavePlots(range(len(decucbCtr_b)), ucbCtr_b, decucbCtr_b, 100000, startTimes, locations ,name)

		
		title("Day:"+day+" binLength:"+binLength+" decay=1-.1e-8")


	# finding the difference of CTR after and before 
	if stats:
		linCTR = summary(articlesLinUCB, 'ucbCTR')
		decayLinCTR = summary(articlesDecayLinUCB, 'ucbCTR')
		linClicks = summary(articlesLinUCB, 'ucbc')
		decayLinClicks = summary(articlesDecayLinUCB, 'ucbc')

		diff = [(str(x[0]), str(y[0]), x[1] - y[1], w[1], z[1]) for x,y,w,z in zip(linCTR, decayLinCTR, linClicks, decayLinClicks)]
		diff = sorted(diff, key = itemgetter(2))
		differences = [x[2] for x in diff]
		hist(differences)
		print '\n'.join([','.join(str(y) for y in x) for x in diff])

	
		# from reset_count find the articles

	if plots_ and analysistype=="decayLinUCB":
		differences = map(lambda x: (float(x[0]), float(x[2])), diff)
		differences = sorted(differences, key = itemgetter(1))
		keys = differences[:20] 
		plotArticleCurves(articlesDecayLinUCB, articlesDecayLinUCB.keys(), decucbStats, keys)

		id = 109576
		objC = articlesDecayLinUCB[id]
		objM = articlesLinUCB[id]

		# tim = np.array(range(shape(articlesSingle[id][0])[0]))
		maxClick = objC.ucbc[-1]
		# calculating batch stats
		with np.errstate(invalid='ignore'):
			objC_access_batch = np.concatenate((np.array([objC.ucba[0]]), objC.ucba[1:] - objC.ucba[:-1]))
			objC_click_batch = np.concatenate((np.array([objC.ucbc[0]]), objC.ucbc[1:] - objC.ucbc[:-1]))
			objC_ctr_batch = objC_click_batch / objC_access_batch

			objM_access_batch = np.concatenate((np.array([objM.ucba[0]]), objM.ucba[1:] - objM.ucba[:-1]))
			objM_click_batch = np.concatenate((np.array([objM.ucbc[0]]), objM.ucbc[1:] - objM.ucbc[:-1]))
			objM_ctr_batch = objM_click_batch / objM_access_batch

			accessMultrandBatches = np.concatenate((np.array([objM.Oranda[0]]), objM.Oranda[1:] - objM.Oranda[:-1]))
			clickMultrandBatches = np.concatenate((np.array([objM.Orandc[0]]), objM.Orandc[1:] - objM.Orandc[:-1]))
			batchMultrandCTR = clickMultrandBatches / accessMultrandBatches

		yAccMax =max([max(objM_access_batch), max(objC_access_batch)])
		yCTRMax =max([max(objM_ctr_batch), max(objC_ctr_batch)])
		thetaMax = np.max([np.max(objC.theta),np.max(objM.theta)])
		thetaMin = np.min([np.min(objC.theta),np.min(objM.theta)])
		minTim, maxTim = min(objC.tim), max(objC.tim)
		if 1:
			f, axarr = plt.subplots(4, sharex=True)
			axarr[0].set_ylim([thetaMin, thetaMax])
			axarr[0].plot(objC.tim, objC.theta)

			axarr[0].set_ylabel('DecayLinUCB \nPreferences')
			# axarr[0].set_title("Fluctuating popularity of article")
			# axarr[0].set_title('title ID : ' + str(id) + 'Sessions:' + ','.join([str(x) for x in list(set(objH.resetCount))]) + ' Days:' + ','.join([str(x) for x in list(set(objS.resetCount))]))

			axarr[1].set_ylim([thetaMin, thetaMax])
			axarr[1].plot(objM.tim, objM.theta)
			axarr[1].set_ylabel('LinUCB \nPreferences')

			# axarr[0].set_title('Plots of Daily Dynamic Theta for Min Diff CTR')
			axarr[2].set_ylim([0, yAccMax])
			axarr[2].plot(objC.tim, objC_access_batch)
			# axarr[2].plot(objC.tim, objS.ucba)
			axarr[2].plot(objM.tim, objM_access_batch, 'm')
			# axarr[2].plot(objM.tim, objM.ucba, 'm')

			axarr[2].set_ylabel('LinUCB Access', color='m')
			axarrT = axarr[2].twinx()
			axarrT.set_ylim([0, yCTRMax])
			axarrT.plot(objC.tim, objC_ctr_batch, 'r')
			axarrT.plot(objM.tim, objM_ctr_batch, 'g')
			axarrT.plot(objM.tim, batchMultrandCTR, 'y')
			axarrT.set_ylabel('LinUCB CTR', color='g')
			axarr[2].legend(['Decay Acc','LinUCB Acc'], loc=2, prop={'size':8})
			axarrT.legend(['Decay CTR','LinUCB CTR'], loc=1, prop={'size':8})
			# axarr[3].plot(objC.tim, objC.inPool, '.')
			# axarr[3].plot(objC.tim, objC.selected, '.')
			# axarr[3].plot(objM.tim, objM.inPool, '.')
			# axarr[3].plot(objM.tim, objM.selected, '.')
			# axarr[3].legend(['In Pool', 'Selected'])
			axarr[3].plot(objC.tim, objC.varClus*alpha, 'b')
			axarr[3].plot(objM.tim, objM.varClus*alpha, 'r')
			axarr[3].legend(['DecayLinUCB','LinUCB'], loc=3, prop={'size':10})
			# axarr[4].plot(objS.tim, objM.means)

			# axarr[2].set_xlabel('title ID : ' + str(id) + ' Sessions:' + ','.join([str(x) for x in list(set(objH.resetCount))]) + ' Days:' + ','.join([str(x) for x in list(set(objS.resetCount))]))
			# xlabel('days:' + ','.join([str(x) for x in list(set(objC.resetCount))]))
			
			for t in objC.restartTimes:
				if t > minTim and t < maxTim:
					xSet = [t for it in range(31)]
					maxCTR = max([max(objC_ctr_batch), max(objM_ctr_batch), max(batchMultrandCTR)])
					
					ySet = thetaMin + (np.array(range(0, 31))*1.0/30)*(thetaMax - thetaMin)
					axarr[0].plot(xSet, ySet, 'black')
					axarr[1].plot(xSet, ySet, 'black')
					ySet = (np.array(range(0, 31))*1.0/30)*maxCTR
					axarrT.plot(xSet, ySet, 'black')


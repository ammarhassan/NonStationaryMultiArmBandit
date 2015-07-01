from conf import * 	# it saves the address of data stored and where to save the data produced by algorithms
import time
import re 			# regular expression library
from random import random, choice 	# for random strategy
from operator import itemgetter  	# for easiness in sorting and finding max and stuff
import datetime
import numpy as np 	# many operations are done in numpy as matrix inverse; for efficiency


# time conventions in file name
# dataDay + Month + Day + Hour + Minute

# data structures for different strategies and parameters

# data structure to store ctr
class articleAccess():
	def __init__(self):
		self.accesses = 0.0 # times the article was chosen to be presented as the best articles
		self.clicks = 0.0 	# of times the article was actually clicked by the user
		self.CTR = 0.0 		# ctr as calculated by the updateCTR function

	def updateCTR(self):
		try:
			self.CTR = self.clicks / self.accesses
		except ZeroDivisionError: # if it has not been accessed
			self.CTR = 0
		return self.CTR

	def addrecord(self, click):
		self.clicks += click
		self.accesses += 1

# structure to save data from random strategy as mentioned in LiHongs paper
class randomStruct:
	def __init__(self):
		self.learn_stats = articleAccess()
		self.deploy_stats = articleAccess()

# data structure for LinUCB for a single article; 
class LinUCBStruct:
	def __init__(self, d, tim = None):
		self.A = np.identity(n=d) 			# as given in the pseudo-code in the paper
		self.b = np.zeros(d) 				# again the b vector from the paper 
		self.A_inv = np.identity(n=d)		# the inverse
		self.learn_stats = articleAccess()	# in paper the evaluation is done on two buckets; so the stats are saved for both of them separately; In this code I am not doing deployment, so the code learns on all examples
		self.deploy_stats = articleAccess()
		self.theta = np.zeros(d)			# the famous theta
		self.pta = 0 						# the probability of this article being chosen
		self.var = 0
		self.mean = 0
		self.DD = np.identity(n=d)*0
		self.identityMatrix = np.identity(n=d)
		self.last_access_time = tim

	def reInitilize(self):
		d = np.shape(self.A)[0]				# as theta is re-initialized some part of the structures are set to zero
		self.A = np.identity(n=d)
		self.b = np.zeros(d)
		self.A_inv = np.identity(n=d)
		self.theta = np.zeros(d)
		self.pta = 0

	def updateTheta(self):
		# self.updateA(featureVector, decay, current_time)
		# self.updateB(featureVector, click, decay, current_time)
		# self.updateInv()
		self.theta = np.dot(self.A_inv, self.b) # as good software code a function to update internal variables

	def updateParameters(self, featureVector, click, decay=None, current_time=None):
		self.updateA(featureVector, decay, current_time)
		self.updateB(featureVector, click, decay, current_time)
		self.updateInv()
		self.updateTheta()
		# self.theta = np.dot(self.A_inv, self.b) # as good software code a function to update internal variables

	def updateInv(self):
		self.A_inv = np.linalg.inv(self.A)		# update the inverse

	def updateA(self, featureVector, decay=None, current_time=None):
		if decay:
			assert decay <= 1 and decay >=0
			self.DD = self.decayAverage(decay, self.DD, np.outer(featureVector, featureVector), current_time)
			self.A = self.DD + self.identityMatrix
		else:
			self.DD +=np.outer(featureVector, featureVector)
			self.A = self.DD + self.identityMatrix

	def updateB(self, featureVector, click, decay=None, current_time=None):
		if decay:
			assert decay <= 1 and decay >= 0
			self.b = self.decayAverage(decay, self.b, featureVector*click, current_time)
		else:
			self.b += featureVector*click

	def applyDecay(self, decay, duration):
		self.DD *= (decay**duration)
		self.A = self.identityMatrix + self.DD
		self.updateInv()		
		self.b *= (decay**duration)
		self.updateTheta()

	def decayAverage(self, decay, previousInsts, newInstance, current_time):
		if current_time:
			results = decay**(current_time - self.last_access_time) * previousInsts + newInstance
			self.last_access_time = current_time
			return results
		return decay*previousInsts + newInstance

	def evaluateStats(self, featureVector, alpha):
		self.mean = np.dot(self.theta, featureVector)
		self.var = np.sqrt(np.dot(np.dot(featureVector,self.A_inv), featureVector))
		self.pta = self.mean + alpha * self.var

	def getSpecialVar(self):
		featureVector = np.ones(len(self.b)) / len(self.b)
		return "{:0.3f}".format(np.sqrt(np.dot(np.dot(featureVector,self.A_inv), featureVector)))

	def getMean(self):
		return self.mean

def simulationTestOfDecay(articles_LinUCB, users, iterations, lastDecay=None):
	print "Simulation Initiated"
	import copy
	n_articles = copy.deepcopy(articles_LinUCB)
	user = choice(users)
	user = users[-1]
	print user
	for key in n_articles:
		n_articles[key].learn_stats.updateCTR()

	ctrs_sorted = sorted(map(
		lambda x: (x,n_articles[x].learn_stats.CTR),
		n_articles.keys()
		), key=itemgetter(1))
	print ctrs_sorted

	twoArticles = ctrs_sorted[-1][0], ctrs_sorted[-2][0]
	# twoArticles = sample(n_articles.keys(), 2)
	if lastDecay:
		decay = lastDecay
	else: decay = 1
	alpha = .3
	# record results
	results = {}
	randCtrs = {}

	for art in twoArticles:
		results[art] = {'mean':[], 'var':[], 'pta':[]}
		randCtrs[art] = .04

	for iteration in xrange(iterations):
		if lastDecay:
			applyDecayToAll(n_articles, decay, 1)
		for art in twoArticles:
			# print art, n_articles[art].DD[0][0], articles_LinUCB[art].DD[0][0]
			n_articles[art].evaluateStats(user, alpha)
			results[art]['mean'].append(n_articles[art].mean)
			results[art]['var'].append(n_articles[art].var)
			results[art]['pta'].append(n_articles[art].pta)

		article_chosen = max([[x, n_articles[x].pta] for x in twoArticles], key = itemgetter(1))[0]
		click = random() < randCtrs[article_chosen]
		n_articles[article_chosen].updateParameters(user, click)
	return results

def plotSimulation(results):
	import matplotlib.pylab
	f, axarr = plt.subplots(3, sharex=True)

	art1, art2 = results.keys()
	time_ = range(len(results[art1]['mean']))
	upper = max(
		map(
			lambda x: max(
				max(results[x]['var']), max(results[x]['mean']), max(results[x]['pta'])
			),results
		)
	)
	# axarr[0].set_ylim([0, upper])
	# axarr[1].set_ylim([0, upper])
	# axarr[2].set_ylim([0, upper])
	axarr[0].plot(time_, results[art1]['var'], time_, results[art2]['var'])
	axarr[1].plot(time_, results[art1]['pta'], time_, results[art2]['pta'])
	axarr[2].plot(time_, results[art1]['mean'], time_, results[art2]['mean'])
	axarr[0].legend(["art1 var", "art2 var"])
	axarr[1].legend(["art1 pta", "art2 pta"])
	axarr[2].legend(["art1 mean", "art2 mean"])


# this is for without context UCB. This is not used in this code. for future implementations
class UCBStruct:								
	def __init__(self):
		self.learn_stats = articleAccess()
		self.deploy_stats = articleAccess()
		self.confInter = 0.0

	def updateConfInter(self, alpha):
		self.confInter = alpha * 1/np.sqrt(self.learn_stats.clicks)

# this structure saves for the e-greedy algorithm
class GreedyStruct:
	def __init__(self):
		self.learn_stats = articleAccess()
		self.deploy_stats = articleAccess()

# class GreedySegStruct():
# 	def __init__(self, numUsers):
# 		self.clusters = dict([(x, GreedyStruct()) for x in range(numUsers)])

# class LinUCBSegStruct():
# 	def __init__(self, numUsers):
# 		self.clusters = dict([(x, UCBStruct()) for x in range(numUsers)])

# This code simply reads one line from the source files of Yahoo!. Please see the yahoo info file to understand the format. I tested this part; so should be good but second pair of eyes could help
def parseLine(line):
	line = line.split("|")

	tim, articleID, click = line[0].strip().split(" ")
	tim, articleID, click = int(tim), int(articleID), int(click)
	# user_features = np.array([x for ind,x in enumerate(re.split(r"[: ]", line[1])) if ind%2==0][1:])
	user_features = np.array([float(x.strip().split(':')[1]) for x in line[1].strip().split(' ')[1:]])

	pool_articles = [l.strip().split(" ") for l in line[2:]]
	pool_articles = np.array([[int(l[0])] + [float(x.split(':')[1]) for x in l[1:]] for l in pool_articles])
	return tim, articleID, click, user_features, pool_articles
	# returns time, id of selected article, if clicked i.e. the response, 

# this code saves different parameters in the file for one batch; this code is written to meet special needs since we need to see statistics as they evolve; I record accumulative stats from which batch stats can be extracted easily
# dicts: is a dictionary of articles UCB structures indexed by 'article-id' key. to reference an article we do dicts[article-id]
# recored_stats: are interesting statistics we want to save; i save accesses and clicks for UCB, random and greedy strategy for all articles in this batch
# epochArticles: are articles that were availabe to be chosen in this epoch or interval or batch.
# epochSelectedArticles: are articles that were selected by any algorithm in this batch.
# tim: is time of the last observation in the batch
def save_to_file(fileNameWrite, dicts, recordedStats, epochArticles, epochSelectedArticles, tim):
	with open(fileNameWrite, 'a+') as f:
		f.write('data') # the observation line starts with data;
		f.write(',' + str(tim))
		f.write(',' + ';'.join([str(x) for x in recordedStats]))
		f.write(',' + ';'.join(['|'.join(str(x) for x in variance_of_clusters(5, dicts[x].A_inv,bias=False)) + ' ' + str(dicts[x].learn_stats.accesses) + ' ' + str(dicts[x].learn_stats.clicks) + ' ' + str(x) + ' ' + '|'.join(["{:0.4f}".format(y) for y in dicts[x].theta]) for x in epochSelectedArticles]))
		f.write(',' + ';'.join(str(x)+' ' + str(epochArticles[x]) for x in epochArticles))
		f.write(',' + ';'.join(str(x)+' ' + str(epochSelectedArticles[x]) for x in epochSelectedArticles))

		f.write('\n')

# this code counts the line in a file; we need to divide data if we are re-setting theta multiple times a day. Could have been done based on time; i guess little harder to implement
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def variance_of_clusters(numberClusters, A_inv, bias=False):
	varI = np.zeros(numberClusters)
	if bias:
		vector = np.zeros(numberClusters + 1)
		vector[numberClusters] = 1
	else:
		vector = np.zeros(numberClusters)

	for i in range(numberClusters):
		vectorI = vector
		vectorI[i] = 1
		varI[i] = np.sqrt(np.dot(np.dot(vectorI, A_inv), vectorI))
	return varI

def cumAccess(dictionary, learn=True):
	if learn:
		return sum([dictionary[x].learn_stats.accesses for x in dictionary])
	else:
		return sum([dictionary[x].deploy_stats.accesses for x in dictionary])
def cumClicks(dictionary, learn=True):
	if learn:
		return sum([dictionary[x].learn_stats.clicks for x in dictionary])
	else:
		return sum([dictionary[x].deploy_stats.clicks for x in dictionary])

def calculateCTRfromDict(dictionary, learn=True):
	accesses = cumAccess(dictionary, learn)
	if accesses:
		return cumClicks(dictionary, learn)*1.0 / accesses
	else: return 0

def parametersFromInt(alpha, decay):
	alpha = .1 * alpha
	decay = pow(decay*.1, 1.0/(24*60*60))
	return alpha, decay

def applyDecayToAll(articles_LinUCB, decay, duration):
	for key in articles_LinUCB:
		articles_LinUCB[key].applyDecay(decay, duration)
	return True


# the first thing that executes in this programme is this main function
if __name__ == '__main__':
	# I regularly print stuff to see if everything is going alright. Its usually helpfull if i change code and see if i did not make obvious mistakes
	# this function is inside main so that it shares variables with main and I dont wanna have large number of function arguments
	def printWrite():
		# here I calculate stats for individual articles
		randomLA = cumAccess(articles_random, learn=True)
		randomC = cumClicks(articles_random, learn=True)
		randomLearnCTR = randomC *1.0 / randomLA 
		
		UCBLA = cumAccess(articles_LinUCB, learn=True) 
		UCBC = cumClicks(articles_LinUCB, learn=True)
		UCBLearnCTR = UCBC * 1.0 / UCBLA 
		
		greedyLA = cumAccess(articles_greedy, learn=True)
		greedyC = cumClicks(articles_greedy, learn=True)
		greedyLearnCTR = greedyC *1.0/ greedyLA
		
		print totalObservations, len(epochSelectedArticles),
		print 'UCBLrn', UCBLearnCTR / randomLearnCTR,
		print 'GreedLrn', greedyLearnCTR / randomLearnCTR,
		if p_learn < 1:
			randomDeployCTR = calculateCTRfromDict(articles_random, learn=False) 
			UCBDeployCTR = calculateCTRfromDict(articles_LinUCB, learn=False) 
			greedyDeployCTR = calculateCTRfromDict(articles_greedy, learn=False) 
		

			print 'UCBDep', UCBDeployCTR / randomDeployCTR,
			print 'GreedDep', greedyDeployCTR / randomDeployCTR,
		print ' '

		recordedStats = [ UCBLA, UCBC, randomLA, randomC, greedyLA, greedyC]
		# write to file
		save_to_file(fileNameWrite, articles_LinUCB, recordedStats, epochArticles, epochSelectedArticles, tim)

	# this function reset theta for all articles
	def re_initialize_article_Structs():
		for x in articles_LinUCB:
			articles_LinUCB[x].reInitilize()


	modes = {0:'multiple', 1:'single', 2:'hours'} 	# the possible modes that this code can be run in; 'multiple' means multiple days or all days so theta dont change; single means it is reset every day; hours is reset after some hours depending on the reInitPerDay. 
	mode = 'multiple' 									# the selected mode
	fileSig = 'multiple'								# depending on further environment parameters a file signature to remember those. for example if theta is set every two hours i can have it '2hours'; for 
	reInitPerDay = 6								# how many times theta is re-initialized per day
	batchSize = 20000								# size of one batch
	testEnviornment = True
	simulation = True

	decay = 1
	d = 5 											# dimension of the input sizes
	alpha_range = [i for i in range(3,4)]
	decay_range = [i for i in range(1,2)]
	# decay_range = [None]
	ucbVars = [] 										# alpha in LinUCB; see pseudo-code
	clustUser = [0 for x in range(5)]
	parameter_sweep = []
	for alpha in alpha_range:
		for decay in decay_range:
			parameter_sweep.append([alpha, decay])

	for alp, dec  in parameter_sweep:

		fileSig = 'Decay='+ str(dec)+'alpha='+str(alp)+mode
		alpha, decay = parametersFromInt(alp, dec)

		if mode=="hours":
			fileSig = fileSig + str(24/reInitPerDay)

		eta = .2 										# parameter in e-greedy algorithm
		p_learn = 1 									# determined the size of learn and deployment bucket. since its 1; deployment bucked is empty

		# respective dictionaries for algorithms
		articles_LinUCB = {} 
		articles_greedy = {}
		articles_random = {}

		ctr = 1 				# overall ctr
		numArticlesChosen = 1 	# overall the articles that are same as for LinUCB and the random strategy that created Yahoo! dataset. I will call it evaluation strategy
		totalObservations = 0 		# total articles seen whether part of evaluation strategy or not
		totalClicks = 0 		# total clicks 
		randomNum = 1 			# random articles chosen
		count = 0 				# not very usefull
		countNoArticle = 0 		# total articles in the pool 
		countLine = 0 			# number of articles in this batch. should be same as batch size; not so usefull
		resetInterval = 0 		# initialize; value assigned later; determined when 
		timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M') 	# the current data time
		last_time = 0
		dataDays = ['03', '04', '05', '06', '07', '08', '09', '10'] # the files from Yahoo that the algorithms will be run on; these files are indexed by days starting from May 1, 2009. this array starts from day 3 as also in the test data in the paper
		
		if testEnviornment:
			dataDays = ['01']

		for dataDay in dataDays:


			# fileName = yahoo_address + "/ydata-fp-td-clicks-v1_0.200905" + dataDay

			fileName = yahoo_address + "/ydata-fp-td-clicks-v1_0.200905" + dataDay
			epochArticles = {} 			# the articles that are present in this batch or epoch
			epochSelectedArticles = {} 	# the articles selected in this epoch
			hours = 0 					# times the theta was reset if the mode is 'hours'		
			batchStartTime = 0 			# time of first observation of the batch

			# should be self explaining
			if mode == 'single':
				fileNameWrite = os.path.join(save_address, fileSig + dataDay + timeRun + '.csv')
				re_initialize_article_Structs()

				countNoArticle = 0

			elif mode == 'multiple':
				fileNameWrite = os.path.join(save_address, fileSig +dataDay + timeRun + '.csv')
			elif mode == 'hours':
				numObs = file_len(fileName)
				# resetInterval calcualtes after how many observations the count should be reset?
				resetInterval = int(numObs / reInitPerDay) + 1
				fileNameWrite = os.path.join(save_address, fileSig + dataDay + '_' + str(hours) + timeRun + '.csv')

			# put some new data in file for readability
			with open(fileNameWrite, 'a+') as f:
				f.write('\nNew Run at  ' + datetime.datetime.now().strftime('%m/%d/%Y %H:%M:%S'))

				# format style, '()' means that it is repeated for each article
				f.write('\n, Time,UCBLearnAccesses;UCBClicks;randomLearnAccesses;randomClicks;greedyLearnAccesses;greedyClicks,(varOfClusterUsers Article_Access Clicks ID Theta),(ID;epochArticles),(ID;epochSelectedArticles)\n')

			print fileName, fileNameWrite, dataDay, resetInterval

			with open(fileName, 'r') as f:
				# reading file line ie observations running one at a time
				for line in f:

					countLine = countLine + 1
					totalObservations = totalObservations + 1

					# if mode is hours and time to reset theta
					if mode=='hours' and countLine > resetInterval:
						hours = hours + 1
						# each time theta is reset, a new file is started.
						fileNameWrite = os.path.join(save_address, fileSig + dataDay + '_' + str(hours) + timeRun + '.csv')
						# re-initialize
						countLine = 0
						re_initialize_article_Structs()
						printWrite()
						batchStartTime = tim
						epochArticles = {}
						epochSelectedArticles = {}
						print "hours thing fired!!"

					# number of observations seen in this batch; reset after start of new batch
					

					# read the observation
					tim, article_chosen, click, user_features, pool_articles = parseLine(line)
					# if tim != last_time:
					# 	applyDecayToAll(articles_LinUCB, decay, tim - last_time)
					# 	last_time = tim
					clustUser[max(enumerate(user_features[:-1]), key=itemgetter(1))[0]]+=1

					# article ids for articles in the current pool for this observation
					currentArticles = []
					for article in pool_articles:
						# featureVector = np.concatenate((user_features[:-1], article[1:-1]), axis = 0)

						# exclude 1 from feature vectors
						featureVector = user_features[:-1]
						# if there is a problem with the feature vector, skip this observation
						if len(featureVector) is not d:
							print 'feature_vector len mismatched'
							continue

						article_id = article[0]
						currentArticles.append(article_id)
						if article_id not in articles_LinUCB: #if its a new article; add it to dictionaries
							articles_LinUCB[article_id] = LinUCBStruct(d, tim)
							articles_greedy[article_id] = GreedyStruct()
							articles_random[article_id] = randomStruct()
							

						if article_id not in epochArticles:
							epochArticles[article_id] = 1
						else:
							# we also count the times article appeared in selection pool in this batch
							epochArticles[article_id] = epochArticles[article_id] + 1

						# Calculate LinUCB confidence bound; done in three steps for readability
						# please check this code for correctness
						articles_LinUCB[article_id].mean = np.dot(articles_LinUCB[article_id].theta, featureVector)
						articles_LinUCB[article_id].var = np.sqrt(np.dot(np.dot(featureVector,articles_LinUCB[article_id].A_inv), featureVector))
						articles_LinUCB[article_id].pta = articles_LinUCB[article_id].mean + alpha * articles_LinUCB[article_id].var
						# articles_LinUCB[article_id].evaluateStats(featureVector, alpha)

					if article_chosen not in epochSelectedArticles:
						epochSelectedArticles[article_chosen] = 1
					else:
						epochSelectedArticles[article_chosen] = epochSelectedArticles[article_chosen] + 1					
						# if articles_LinUCB[article_id].pta < 0: print 'PTA', articles_LinUCB[article_id].pta,

					# articles picked by LinUCB
					ucbArticle = max(np.random.permutation([(x, articles_LinUCB[x].pta) for x in currentArticles]), key=itemgetter(1))[0]

					# article picked by random strategy
					randomArticle = choice(currentArticles)

					# article picked by greedy
					greedyArticle = max([(x, articles_greedy[x].learn_stats.CTR) for x in currentArticles], key = itemgetter(1))[0]
					if random() < eta: greedyArticle = choice(currentArticles)

					learn = random()<p_learn # decide the learning or deployment bucket

					# if random strategy article Picked by evaluation srategy
					if randomArticle == article_chosen:
						if learn:
							articles_random[randomArticle].learn_stats.addrecord(click)
						else:
							articles_random[randomArticle].deploy_stats.addrecord(click)

					# if LinUCB article is the chosen by evaluation strategy; update datastructure with results
					if ucbArticle==article_chosen:
						if learn: # if learning bucket then use the observation to update the parameters
							articles_LinUCB[article_chosen].learn_stats.addrecord(click)

							articles_LinUCB[article_chosen].updateA(featureVector)
							articles_LinUCB[article_chosen].updateB(featureVector, click)
							articles_LinUCB[article_chosen].updateInv()

							articles_LinUCB[article_chosen].updateTheta()
							# articles_LinUCB[article_chosen].updateParameters(featureVector, click)
						else:
							articles_LinUCB[article_chosen].deploy_stats.addrecord(click)

					# if greedy article is chosen by evalution strategy
					# if greedyArticle == article_chosen:
					# 	if learn:
					# 		articles_greedy[article_chosen].learn_stats.addrecord(click)
					# 		articles_greedy[article_chosen].learn_stats.updateCTR()
					# 	else:
					# 		articles_greedy[article_chosen].deploy_stats.addrecord(click)
							
						
					# if the batch has ended 
					if totalObservations%batchSize==0:

						# write observations for this batch
						printWrite()
						
						

						batchStartTime = tim
						epochArticles = {}
						epochSelectedArticles = {}

					# if article_chosen==ucbArticle:
					# 	# print article_chosen, [(_, "{:0.3f}".format(articles_LinUCB[_].getMean()), articles_LinUCB[_].getSpecialVar()) for _ in articles_LinUCB.keys() if articles_LinUCB[_].getMean()>0]
					# 	print article_chosen, click, np.linalg.det(np.outer(featureVector, featureVector)), [(_, np.linalg.det(articles_LinUCB[_].A)) for _ in articles_LinUCB.keys() if articles_LinUCB[_].getMean()>0 or _==article_chosen] 
					# 	ucbVars.append(articles_LinUCB[109498].getSpecialVar())

					totalClicks = totalClicks + click

					if simulation and testEnviornment and totalObservations> 10000:
						try:
							test_user_features.append(featureVector)
						except:
							test_user_features = []
						if totalObservations > 10100:
							print "Its gonna break!!"
							break

					# if totalObservations > 1000:
						# break

				# print stuff to screen and save parameters to file when the Yahoo! dataset file ends
				printWrite()

				
	if simulation and testEnviornment:
		
		results = simulationTestOfDecay(articles_LinUCB, test_user_features, iterations=2000, lastDecay=.9)
		plotSimulation(results)



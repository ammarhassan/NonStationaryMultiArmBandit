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
class LinUCBStruct(object):
	def __init__(self, d, articleID, alpha,tim = None):
		self.articleID = articleID
		self.A = np.identity(n=d) 			# as given in the pseudo-code in the paper
                self.A_inv= np.linalg.inv(self.A)
		self.b = np.zeros(d) 				# again the b vector from the paper 
		self.alpha = alpha
		
		self.theta = np.dot(self.A_inv, self.b)
		
		self.pta = 0 						# the probability of this article being chosen
		self.var = 0
		self.mean = 0

		self.identityMatrix = np.identity(n=d)

		self.learn_stats = articleAccess()	# in paper the evaluation is done on two buckets; so the stats are saved for both of them separately; In this code I am not doing deployment, so the code learns on all examples
		self.deploy_stats = articleAccess()
		
		self.last_access_time = tim
		

	def reInitilize(self):
		d = np.shape(self.A)[0]				# as theta is re-initialized some part of the structures are set to zero
		self.A = np.identity(n=d)
		self.b = np.zeros(d)
		self.A_inv = np.identity(n=d)
		self.theta = np.dot(self.A_inv, self.b)

	def updateParameters(self, featureVector, click):
		self.A += np.outer(featureVector, featureVector)
		self.b += featureVector*click
		self.A_inv = np.linalg.inv(self.A)

		self.theta = np.dot(self.A_inv, self.b)

	def getProb(self,featureVector):
		self.mean = np.dot(self.theta, featureVector)
		self.var =  np.sqrt(np.dot(np.dot(featureVector,self.A_inv ), featureVector))
		self.pta = self.mean + self.alpha * self.var



class Researt_LinStruct(LinUCBStruct):
	def __init__(self, d, articleID, alpha, tim = None):
		LinUCBStruct.__init__(self, d =d , articleID = articleID, alpha =alpha, tim = tim)
		self.counter = 0
		self.intervalNum = 0

class Decay_LinStruct(LinUCBStruct):
	def __init__(self, d, articleID, alpha, belta, tim = None):
		LinUCBStruct.__init__(self, d =d , articleID = articleID, alpha =alpha, tim = tim)
		self.belta = belta
		self.last_access_time = tim
	def updateParameters(self, featureVector, click):
		pass

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
	#print 'save'
    
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
	decay = 1 - pow(.1, decay)
	return alpha, decay

def applyDecayToAll(articles_ReStartLinUCB, decay, duration):
	for key in articles_ReStartLinUCB:
		articles_ReStartLinUCB[key].applyDecay(decay, duration)
	return True


# the first thing that executes in this programme is this main function
if __name__ == '__main__':
	# I regularly print stuff to see if everything is going alright. Its usually helpfull if i change code and see if i did not make obvious mistakes
	# this function is inside main so that it shares variables with main and I dont wanna have large number of function arguments
	def printWrite():
		# here I calculate stats for individual articles
		randomLA = cumAccess(articles_random, learn=True)
		randomC = cumClicks(articles_random, learn=True)
		if randomLA == 0:
			randomLearnCTR = 0.0
		else:
			randomLearnCTR = randomC *1.0 / randomLA 
		
		Restart_LinUCBLA = cumAccess(articles_ReStartLinUCB, learn=True) 
		Restart_LinUCBC = cumClicks(articles_ReStartLinUCB, learn=True)
		if Restart_LinUCBLA == 0:
			Restart_LinUCBLearnCTR = 0.0
		else:
			Restart_LinUCBLearnCTR = Restart_LinUCBC * 1.0 / Restart_LinUCBLA 

		LinUCBLA = cumAccess(articles_LinUCB, learn=True) 
		LinUCBC = cumClicks(articles_LinUCB, learn=True)
		if LinUCBLA == 0:
			LinUCBLearnCTR = 0.0
		else:
			LinUCBLearnCTR = LinUCBC * 1.0 / LinUCBLA 
		
			
		print totalObservations, len(epochSelectedArticles),
		print 'Restart_LinUCBLrn', Restart_LinUCBLearnCTR,
		print 'LinUCB', LinUCBLearnCTR
		

		recordedStats = [ Restart_LinUCBLA, Restart_LinUCBC, randomLA, randomC, LinUCBLA, LinUCBC]
		# write to file
		save_to_file(fileNameWrite, articles_ReStartLinUCB, recordedStats, epochArticles, epochSelectedArticles, tim)

	# this function reset theta for all articles
	def re_initialize_article_Structs():
		for x in articles_ReStartLinUCB:
			articles_ReStartLinUCB[x].reInitilize()

	def doublingRestart():
		for x in articles_ReStartLinUCB:
			articles_ReStartLinUCB[x].counter +=1
			if articles_ReStartLinUCB[x].counter >= 10000  and articles_ReStartLinUCB[x].counter == 10000*(2**(articles_ReStartLinUCB[x].intervalNum)):
				articles_ReStartLinUCB[x].reInitilize()
				articles_ReStartLinUCB[x].intervalNum +=1
				print 'reInitilize', str(x)
	def LinearRestart():
		for x in articles_ReStartLinUCB:
                        articles_ReStartLinUCB[x].counter +=1
                        if articles_ReStartLinUCB[x].counter >=1000000 and articles_ReStartLinUCB[x].counter == 1000000 *(1+ articles_ReStartLinUCB[x].intervalNum):
                                articles_ReStartLinUCB[x].reInitilize()
                                articles_ReStartLinUCB[x].intervalNum +=1
                                print 'Linear ReInitilize', str(x)

        modes = {0:'multiple', 1:'single', 2:'hours'} 	# the possible modes that this code can be run in; 'multiple' means multiple days or all days so theta dont change; single means it is reset every day; hours is reset after some hours depending on the reInitPerDay. 
	mode = 'single' 									# the selected mode
	fileSig = 'singleDay'								# depending on further environment parameters a file signature to remember those. for example if theta is set every two hours i can have it '2hours'; for 
	reInitPerDay = 6								# how many times theta is re-initialized per day
	batchSize = 1000								# size of one batch
	testEnviornment = False

	decay = 1
	d = 5 											# dimension of the input sizes
	alpha_range = [i for i in range(3,4)]
	decay_range = [i for i in range(8,9)]
	# decay_range = [None]
	ucbVars = [] 										# alpha in LinUCB; see pseudo-code
	parameter_sweep = []
	for alpha in alpha_range:
		for decay in decay_range:
			parameter_sweep.append([alpha, decay])

	#for alp, dec  in parameter_sweep:

	alpha = 0.3
	decay = 0.99
	#alpha, decay = parametersFromInt(alp, dec)

	if mode=="hours":
		fileSig = fileSig + str(24/reInitPerDay)

	eta = .2 										# parameter in e-greedy algorithm
	p_learn = 1 									# determined the size of learn and deployment bucket. since its 1; deployment bucked is empty

	# respective dictionaries for algorithms
	articles_ReStartLinUCB = {} 
	articles_DecayLinUCB = {}
	articles_LinUCB = {}
	articles_random = {}


	ctr = 1 				# overall ctr
	numArticlesChosen = 1 	# overall the articles that are same as for LinUCB and the random strategy that created Yahoo! dataset. I will call it evaluation strategy
	totalObservations = 0 		# total articles seen whether part of evaluation strategy or not
	effectiveObservations = 0
	totalClicks = 0 		# total clicks 
	randomNum = 1 			# random articles chosen
	count = 0 				# not very usefull
	countNoArticle = 0 		# total articles in the pool 
	countLine = 0 			# number of articles in this batch. should be same as batch size; not so usefull
	resetInterval = 0 		# initialize; value assigned later; determined when 
	timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M') 	# the current data time
	last_time = 0
	dataDays = ['03','04','05','06', '07', '08', '09', '10'] # the files from Yahoo that the algorithms will be run on; these files are indexed by days starting from May 1, 2009. this array starts from day 3 as also in the test data in the paper
	
	if testEnviornment:
		dataDays = ['01']
	j = 0

	for dataDay in dataDays:


		# fileName = yahoo_address + "/ydata-fp-td-clicks-v1_0.200905" + dataDay

		fileName = datasets_address + "/ydata-fp-td-clicks-v1_0.200905" + dataDay
		epochArticles = {} 			# the articles that are present in this batch or epoch
		epochSelectedArticles = {} 	# the articles selected in this epoch
		hours = 0 					# times the theta was reset if the mode is 'hours'		
		batchStartTime = 0 			# time of first observation of the batch

		# should be self explaining
		if mode == 'single':
			fileNameWrite = os.path.join(save_address, fileSig + dataDay + timeRun + '.csv')
			re_initialize_article_Structs()
                        						                     
			#re_initializeLinUCB()

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
			f.write('\n, Time,Restart_LinUCBLearnAccesses;Restart_LinUCBClicks;randomLearnAccesses;randomClicks;LinUCBLearnAccesses;LinUCBClicks,(varOfClusterUsers Article_Access Clicks ID Theta),(ID;epochArticles),(ID;epochSelectedArticles)\n')

		print fileName, fileNameWrite, dataDay, resetInterval
		timcount = 0
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
                                        #doublingRestart() 
					printWrite()
					batchStartTime = tim
					epochArticles = {}
					epochSelectedArticles = {}
					print "hours thing fired!!"

				# number of observations seen in this batch; reset after start of new batch
				

				# read the observation
				tim, article_chosen, click, user_features, pool_articles = parseLine(line)
			
				currentArticles = []
				for article in pool_articles:
					featureVector = user_features[:-1]
					# if there is a problem with the feature vector, skip this observation
					if len(featureVector) is not d:
						print 'feature_vector len mismatched'
						continue

					article_id = article[0]
					currentArticles.append(article_id)
					if article_id not in articles_ReStartLinUCB: #if its a new article; add it to dictionaries
						articles_ReStartLinUCB[article_id] = Researt_LinStruct(d, article_id, alpha, tim)
						articles_LinUCB[article_id] = LinUCBStruct(d,article_id,alpha,tim)
						articles_random[article_id] = randomStruct()
                                        #Doubling restart
                                        #doublingRestart()
                                        #LinearRestart
                                        #LinearRestart()

                                        articles_LinUCB[article_id].getProb(featureVector)
					articles_ReStartLinUCB[article_id].getProb(featureVector)

					# Doubling restart
					#doublingRestart()
                                        

					if article_id not in epochArticles:
						epochArticles[article_id] = 1
					else:
						# we also count the times article appeared in selection pool in this batch
						epochArticles[article_id] = epochArticles[article_id] + 1

					#Decay environment
					#articles_ReStartLinUCB[article_id].pta = articles_ReStartLinUCB[article_id].mean + alpha * (decay**effectiveObservations)*articles_ReStartLinUCB[article_id].var

				if article_chosen not in epochSelectedArticles:
					epochSelectedArticles[article_chosen] = 1
				else:
					epochSelectedArticles[article_chosen] = epochSelectedArticles[article_chosen] + 1					
					# if articles_ReStartLinUCB[article_id].pta < 0: print 'PTA', articles_ReStartLinUCB[article_id].pta,

				# articles picked by Restart_LinUCB
				Restart_LinUCBArticle = max(np.random.permutation([(x, articles_ReStartLinUCB[x].pta) for x in currentArticles]), key=itemgetter(1))[0]

				# articles picked by LinUCB
				LinUCBArticle = max(np.random.permutation([(x, articles_LinUCB[x].pta) for x in currentArticles]), key=itemgetter(1))[0]

				# article picked by random strategy
				randomArticle = choice(currentArticles)

				learn = random()<p_learn # decide the learning or deployment bucket

				# if random strategy article Picked by evaluation srategy
				if randomArticle == article_chosen:
					if learn:
						articles_random[randomArticle].learn_stats.addrecord(click)
					else:
						articles_random[randomArticle].deploy_stats.addrecord(click)

				# if Restart_LinUCB article is the chosen by evaluation strategy; update datastructure with results
				if Restart_LinUCBArticle ==article_chosen:
					if learn: # if learning bucket then use the observation to update the parameters
						articles_ReStartLinUCB[article_chosen].learn_stats.addrecord(click)
						articles_ReStartLinUCB[article_chosen].updateParameters(featureVector, click)

					else:
						articles_ReStartLinUCB[article_chosen].deploy_stats.addrecord(click)

				# if LinUCB article is the chosen by evaluation strategy; update datastructure with results
				if LinUCBArticle ==article_chosen:
					
					if learn: # if learning bucket then use the observation to update the parameters
						articles_LinUCB[article_chosen].learn_stats.addrecord(click)
						articles_LinUCB[article_chosen].updateParameters(featureVector, click)
					else:
						articles_LinUCB[article_chosen].deploy_stats.addrecord(click)
						
					
				# if the batch has ended 
				if totalObservations%batchSize==0:

					# write observations for this batch
					printWrite()
				
					batchStartTime = tim
					epochArticles = {}
					epochSelectedArticles = {}

				# if article_chosen==ucbArticle:
				# 	# print article_chosen, [(_, "{:0.3f}".format(articles_ReStartLinUCB[_].getMean()), articles_ReStartLinUCB[_].getSpecialVar()) for _ in articles_ReStartLinUCB.keys() if articles_ReStartLinUCB[_].getMean()>0]
				# 	print article_chosen, click, np.linalg.det(np.outer(featureVector, featureVector)), [(_, np.linalg.det(articles_ReStartLinUCB[_].A)) for _ in articles_ReStartLinUCB.keys() if articles_ReStartLinUCB[_].getMean()>0 or _==article_chosen] 
				# 	ucbVars.append(articles_ReStartLinUCB[109498].getSpecialVar())

				totalClicks = totalClicks + click

				# if totalObservations > 1000:
					# break

			# print stuff to screen and save parameters to file when the Yahoo! dataset file ends
			printWrite()

		print timcount

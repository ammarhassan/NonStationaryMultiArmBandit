from conf import * 	# it saves the address of data stored and where to save the data produced by algorithms
import time
import re 			# regular expression library
from random import random, choice 	# for random strategy
from operator import itemgetter  	# for easiness in sorting and finding max and stuff
import datetime
import numpy as np 	# many operations are done in numpy as matrix inverse; for efficiency


# time conventions in file name
# dataDay + Month + Day + Hour + Minute


class user():
	def __init__(self, num, referenceVector=None, seen_limit=None):
		# self.referenceVector = self.produce_referenceVector(referenceVector)
		self.referenceVector = None
		self.clicks = {}
		self.accesses = {}
		self.conditionalCTRS = {}

		self.difference = .1
		self.file_ = os.path.join(os.path.join(save_address, "conditionalClicks"), "user%d_cond_func.csv"%num)
		self.seen_limit = seen_limit
		self.user_apearance = 0
		self.data_clicks = []
		self.data_access = []
		self.user_apearance_batches = []
		

	def produce_referenceVector(self, referenceVector):
		if referenceVector is not None: return referenceVector
		ref = np.array([random() for _ in range(5)])
		return ref/sum(ref)

	def addConditionalCTR(self, featureVector, article, click):
		if self.referenceVector is None:
			if random() < 1.0 / self.seen_limit:
				self.referenceVector = featureVector
				print "number %s assigned"%self.referenceVector
				# with open(self.file_, 'a') as f:
				# 	f.write("User," + ','.join(map(str, self.referenceVector)) + '\n')
			return

		if np.linalg.norm(featureVector - self.referenceVector) < self.difference:
			self.user_apearance += 1
			if article in self.clicks:
				# print "click", click
				self.clicks[article] += click
				self.accesses[article] += 1
			else:
				self.clicks[article] = click
				self.accesses[article] = 1

	def writeConditionalCTR(self):
		# with open(self.file_, 'a') as f:
		# 	f.write(','.join(map(lambda x:str(x)+";"+str(self.conditionalCTRS[x]), self.conditionalCTRS)) + '\n')
		self.data_clicks.append(self.clicks)
		self.data_access.append(self.accesses)
		self.user_apearance_batches.append(self.user_apearance)

	def re_initialize(self):
		self.writeConditionalCTR()
		self.clicks = {}
		self.accesses = {}
		self.user_apearance = 0

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

# the first thing that executes in this programme is this main function
if __name__ == '__main__':
	# I regularly print stuff to see if everything is going alright. Its usually helpfull if i change code and see if i did not make obvious mistakes
	# this function is inside main so that it shares variables with main and I dont wanna have large number of function arguments
	def printWrite():
		print "Articles Clicked", sum(
			map(
				lambda x:sum(map(
							lambda y: y[1], x.accesses.items()))
				, random_users))

		for x in random_users:
			x.re_initialize()

	batchSize = 100000								# size of one batch

# 	testEnviornment = True
# 	simulation = True
# 	ucb = 0
# 	greedy = 0


# 	d = 5 											# dimension of the input sizes
# 	# decay_range = [None]
# 	ucbVars = [] 										# alpha in LinUCB; see pseudo-code
# 	clustUser = [0 for x in range(5)]

# 	eta = .2 										# parameter in e-greedy algorithm
# 	p_learn = 1 									# determined the size of learn and deployment bucket. since its 1; deployment bucked is empty

# 	random_users = [user(i, seen_limit=2000) for i in range(20)]
# 	# for i in range(5):
# 	# 	temp = np.zeros(5)
# 	# 	temp[i] = 1
# 	# 	random_users.append(user(i, temp))


# 	ctr = 1 				# overall ctr
# 	numArticlesChosen = 1 	# overall the articles that are same as for LinUCB and the random strategy that created Yahoo! dataset. I will call it evaluation strategy
# 	totalObservations = 0 		# total articles seen whether part of evaluation strategy or not
# 	totalClicks = 0 		# total clicks 
# 	randomNum = 1 			# random articles chosen
# 	count = 0 				# not very usefull
# 	countNoArticle = 0 		# total articles in the pool 
# 	countLine = 0 			# number of articles in this batch. should be same as batch size; not so usefull
# 	resetInterval = 0 		# initialize; value assigned later; determined when 
# 	timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M') 	# the current data time
# 	last_time = 0
# 	dataDays = ['03', '04', '05', '06', '07', '08', '09', '10'] # the files from Yahoo that the algorithms will be run on; these files are indexed by days starting from May 1, 2009. this array starts from day 3 as also in the test data in the paper
	
# 	if testEnviornment:
# 		dataDays = ['01']

# 	for dataDay in dataDays:

# 		# fileName = yahoo_address + "/ydata-fp-td-clicks-v1_0.200905" + dataDay

# 		fileName = yahoo_address + "/ydata-fp-td-clicks-v1_0.200905" + dataDay
# 		epochArticles = {} 			# the articles that are present in this batch or epoch
# 		epochSelectedArticles = {} 	# the articles selected in this epoch
# 		batchStartTime = 0 			# time of first observation of the batch

# 		# should be self explaining

# 		with open(fileName, 'r') as f:
# 			# reading file line ie observations running one at a time
# 			for line in f:

# 				countLine = countLine + 1
# 				totalObservations = totalObservations + 1

# 				# number of observations seen in this batch; reset after start of new batch

# 				# read the observation
# 				tim, article_chosen, click, user_features, pool_articles = parseLine(line)
# 				for u in random_users:
# 					u.addConditionalCTR(user_features[:-1], article_chosen, click)

# 				# if the batch has ended 
# 				if totalObservations%batchSize==0:
# 					# write observations for this batch
# 					printWrite()

# 				totalClicks = totalClicks + click
# 		# print stuff to screen and save parameters to file when the Yahoo! dataset file ends
# 			printWrite()

# # 			if simulation and testEnviornment and totalObservations> 50000:
# # 				try:
# # 					user_features.append(featureVector)
# # 				except:
# # 					user_features = []
# # 				if totalObservations > 50200:
# # 					break
# 	# if simulation and testEnviornment:
# 	# 	from matplotlib.pylab import *
# 	# 	results = simulationTestOfDecay(articles_LinUCB, user_features)
# 	# 	art1, art2 = results.keys()
# 	# 	time_ = range(results[art1]['mean'])
# 	# 	plot(time_, results[art1]['var'], time_, results[art2]['var'])
# 	# 	plot(time_, results[art1]['mean'], time_, results[art2]['mean'])
# 	# 	plot(time_, results[art1]['pta'], time_, results[art2]['pta'])
# 	# 	legend(["art1 var", "art2var", "art1 mean", "art2 mean", "art1 pta", "art2 pta"])


def simulationTestOfDecay(articles_LinUCB, users):
	user = choice(users)
	twoArticles = sample(articles_LinUCB.keys(), 2)
	decay = .999
	# record results
	results = {}
	randCtrs = [.5, .5]

	for art in twoArticles:
		results[art] = {'mean':[], 'var':[], 'pta':[]}

	for iteration in range(10000):

		applyDecayToAll(articles_LinUCB, decay, 1)
		for art in twoArticles:
			articles_LinUCB[art].evaluateStats(user, .3)
			results[art]['mean'].append(articles_LinUCB[art].mean)
			results[art]['var'].append(articles_LinUCB[art].var)
			results[art]['pta'].append(articles_LinUCB[art].pta)

		article_chosen = max([[x, articles_LinUCB[x]] for x in twoArticles], key = itemgetter(1))[0]
		articles_LinUCB[article_chosen].updateTheta(user, choice([0,1]))
	return results

def analyze_user(data, top_a, type):
	popular_articles = set([])
	for d in data:
		temp =map(lambda x: x[0], sorted(d.items(), key=itemgetter(1)))[-top_a:]
		popular_articles = popular_articles.union(set(temp))
	popular_articles = list(popular_articles)
	print popular_articles

	data_array = []
	for d in data:
		temp = [d[id] if id in d else 0 for id in popular_articles]
		if type=="CTR":
			rest = [d[id] for id in d if d not in popular_articles]
			rest = sum(rest) / len(rest)
		elif type=="clicks":
			rest = sum([d[id] for id in d if d not in popular_articles])
		temp.append(rest)
		data_array.append(temp)

	return data_array

def getCTRS(dictsaccess, dictsclicks, size, type):

	numDicts = len(dictsaccess) / size + 1
	summed_dicts_list = []
	
	for i in range(numDicts):
		tempDicta = {}
		tempDictc = {}
		for dicta, dictc in zip(dictsaccess[(i*size):((i+1)*size)], dictsclicks[(i*size):((i+1)*size)]):
			for key in dicta:
				if key in tempDicta:
					tempDicta[key] += dicta[key]
					tempDictc[key] += dictc[key]
				else:
					tempDicta[key] = dicta[key]
					tempDictc[key] = dictc[key]
		tempDict = dict([(key, tempDictc[key]*1.0/tempDicta[key]) for key in tempDicta])
		if type=="CTR":
			summed_dicts_list.append(tempDict)
		elif type=="clicks":
			summed_dicts_list.append(tempDictc)
	return summed_dicts_list

def plotBar(data_array):

	import matplotlib.cm as cmx
	import matplotlib.colors as colors

	def get_cmap(N):
		'''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
		RGB color.'''
		color_norm  = colors.Normalize(vmin=0, vmax=N-1)
		scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
		def map_index_to_rgb_color(index):
		    return scalar_map.to_rgba(index)
		return map_index_to_rgb_color

	a = get_cmap(len(data_array)+1)
	color = [a(i) for i in range(len(data_array))]
	# shuffle(color)
	y = [0 for i in range(len(data_array[0]))]
	for i, x in enumerate(data_array):
		bar(range(len(x)), x, bottom=y, color = color[i])
		y = map(sum, zip(x,y))
		# print color[i], x, y

def plotCharts(random_users, idnum, step, top_a, type):
	d = getCTRS(random_users[idnum].data_access, random_users[idnum].data_clicks, step, type)
	dd = analyze_user(d, top_a, type)
	plotBar(zip(*dd))
	figure()
	bar(range(len(random_users[idnum].user_apearance_batches)),
		random_users[idnum].user_apearance_batches)

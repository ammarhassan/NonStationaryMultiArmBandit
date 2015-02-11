import math
import numpy as np
import random
from MAB_algorithms import *

def evolveTheta(type="linearTimeEvolution", **kwargs):
	if type=="linearTimeEvolution":
		return kwargs["initialTheta"] + (kwargs["initialTheta"] - kwargs["finalTheta"]) * kwargs["curr_time"]*1.0/kwargs["total_iterations"]
	elif type=="constant":
		return kwargs["initialTheta"]

class Article():
	def __init__(self, id, startTime, endTime, FV):
		self.id = id
		self.startTime = startTime
		self.endTime = endTime
		self.initialTheta = None
		self.theta = None
		self.featureVector = FV
		self.deltaTheta = None

	def setTheta(self, theta):
		self.initialTheta = theta
		self.theta = theta

	def setDeltaTheta(self, finalTheta, total_iterations):
		self.deltaTheta = (finalTheta - self.initialTheta) / total_iterations

	def evolveThetaWTime(self):
		self.theta += self.deltaTheta

	def inPool(self, curr_time):
		return curr_time <= self.endTime and curr_time >= self.startTime


class User():
	def __init__(self, id, featureVector):
		self.id = id
		self.featureVector = featureVector

class simulateOnlineData():
	def __init__(self, n_articles, n_users, dimension):
		self.dimension = dimension
		self.articles = []
		self.users = []
		self.simulateArticlePool(n_articles)
		self.simulateUsers(n_users)

	def createIds(self, maxNum):
		return map(
			lambda x:10**math.ceil(math.log(maxNum+1, 10))+x,range(maxNum)
			)

	def featureUniform(self):
		feature = np.array([random() for _ in range(self.dimension)])
		return feature / self.dimension

	def simulateArticlePool(self, n_articles):
		articles_id = self.createIds(n_articles)
		startTimes = [np.random.normal(loc=308023, scale=200430) for x in articles_id]
		minT = min(startTimes)
		startTimes = map(lambda x: x-minT, startTimes)
		durations = [np.random.normal(loc=67433, scale=42674) for x in articles_id]

		for key, st, dur in zip(articles_id, startTimes, durations):
			self.articles.append(Article(key, st, st+dur, self.featureUniform()))
			self.articles[-1].theta = self.featureUniform()

	def simulateUsers(self, numUsers):
		"""users of all context arriving uniformly"""
		usersids = self.createIds(numUsers)
		for key in usersids:
			self.users.append(User(key, self.featureUniform()))

	def evolveTheta(self):
		for x in self.articles:
			x.evolveTheta()

	def runAlgorithms(self, algorithms, time_):
		poolArticles = [x for x in self.articles if x.inPool(time_)]
		userArrived = choice(self.users)
		for alg in algorithms:
			pickedArticle = alg.decide(poolArticles, userArrived, time_)
			clickExpectation = np.dot(pickedArticle.theta, userArrived.featureVector)
			click = np.random.binomial(1, clickExpectation)
			alg.updateParameters(pickedArticle, userArrived, click)
			print "article Picked", pickedArticle.id, "user", userArrived.id, "click", click, "expected_click", clickExpectation

if __name__ == '__main__':

	simExperiment = simulateOnlineData(n_articles=200, n_users=100, dimension=5)

	LinUCB = LinUCBAlgorithm(dimension=5, alpha=.3)

	for i in range(100):
		simExperiment.runAlgorithms([LinUCB], i)


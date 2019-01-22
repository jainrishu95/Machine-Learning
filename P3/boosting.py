import numpy as np
from typing import List, Set
import sys

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
  # Boosting from pre-defined classifiers
	def __init__(self, clfs: Set[Classifier], T=0):
		self.clfs = clfs      # set of weak classifiers to be considered
		self.num_clf = len(clfs)
		if T < 1:
			self.T = self.num_clf
		else:
			self.T = T

		self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
		self.betas = []       # list of weights beta_t for t=0,...,T-1
		return

	@abstractmethod
	def train(self, features: List[List[float]], labels: List[int]):
		return

	def predict(self, features: List[List[float]]) -> List[int]:
		'''
		Inputs:
		- features: the features of all test examples
		Returns:
		- the prediction (-1 or +1) for each example (in a list)
		'''
		########################################################
		# TODO: implement "predict"
		########################################################
		pred = np.zeros(len(features))
		for i in range(self.T):
			pred += self.betas[i] * np.array(self.clfs_picked[i].predict(features))
		print(len(self.clfs_picked), self.betas, np.sign(pred))
		return list(np.sign(pred))

class AdaBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "AdaBoost"
		return

	def train(self, features: List[List[float]], labels: List[int]):
		'''
		Inputs:
		- features: the features of all examples
		- labels: the label of all examples
		Require:
		- store what you learn in self.clfs_picked and self.betas
		'''
		############################################################
		# TODO: implement "train"
		############################################################
		D = [1/len(features) for i in range(len(features))]
		for t in range(self.T):
			min_indicator = sys.maxsize
			min_classifier = None
			min_stump = list()
			for clf in self.clfs:
				stumb = clf.predict(features)
				indicator = 0
				for i in range(len(stumb)):
					if stumb[i] != labels[i]:
						indicator += D[i]
				if min_indicator > indicator:
					min_indicator = indicator
					min_classifier = clf
					min_stump = stumb

			self.clfs_picked.append(min_classifier)
			e_t = min_indicator
			temp = (1 - e_t) / e_t
			b_t = np.log(temp)/2
			self.betas.append(b_t)
			for i in range(len(features)):
				if labels[i] == min_stump[i]:
					D[i] *= np.exp(-b_t)
				else:
					D[i] *= np.exp(b_t)
			sum_ = sum(D)
			for i in range(len(D)):
				D[i] /= sum_

	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)
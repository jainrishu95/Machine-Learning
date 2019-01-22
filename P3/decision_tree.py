import numpy as np
from typing import List
from classifier import Classifier

class DecisionTree(Classifier):
	def __init__(self):
		self.clf_name = "DecisionTree"
		self.root_node = None

	def train(self, features: List[List[float]], labels: List[int]):
		# init.
		assert(len(features) > 0)
		self.feautre_dim = len(features[0])
		num_cls = np.max(labels)+1

		# build the tree
		self.root_node = TreeNode(features, labels, num_cls)
		if self.root_node.splittable:
			self.root_node.split()

		return
		
	def predict(self, features: List[List[float]]) -> List[int]:
		y_pred = []
		for feature in features:
			y_pred.append(self.root_node.predict(feature))
		return y_pred

	def print_tree(self, node=None, name='node 0', indent=''):
		if node is None:
			node = self.root_node
		print(name + '{')
		
		string = ''
		for idx_cls in range(node.num_cls):
			string += str(node.labels.count(idx_cls)) + ' '
		print(indent + ' num of sample / cls: ' + string)

		if node.splittable:
			print(indent + '  split by dim {:d}'.format(node.dim_split))
			for idx_child, child in enumerate(node.children):
				self.print_tree(node=child, name= '  '+name+'/'+str(idx_child), indent=indent+'  ')
		else:
			print(indent + '  cls', node.cls_max)
		print(indent+'}')


class TreeNode(object):
	def __init__(self, features: List[List[float]], labels: List[int], num_cls: int):
		self.features = features
		self.labels = labels
		self.children = []
		self.num_cls = num_cls

		count_max = 0
		for label in np.unique(labels):
			if self.labels.count(label) > count_max:
				count_max = labels.count(label)
				self.cls_max = label # majority of current node

		if len(np.unique(labels)) < 2:
			self.splittable = False
		else:
			self.splittable = True

		self.dim_split = None # the index of the feature to be split

		self.feature_uniq_split = None # the possible unique values of the feature to be split


	def split(self):

		def conditional_entropy(branches: List[List[int]]) -> float:
			'''
			branches: C x B array, 
					  C is the number of classes,
					  B is the number of branches
					  it stores the number of 
					  corresponding training samples 
					  e.g.
					              ○ ○ ○ ○
					              ● ● ● ●
					            ┏━━━━┻━━━━┓
				               ○ ○       ○ ○
				               ● ● ● ●
				               
				      branches = [[2,2], [4,0]]
			'''
			########################################################
			# TODO: compute the conditional entropy
			########################################################
			branches = np.asarray(branches)
			total_in_branch = np.sum(branches, axis=0)
			overall_total = np.sum(total_in_branch)
			branches = branches / total_in_branch
			branches *= np.log2(branches, where=(branches!=0)) * -1
			branches = np.sum(branches, axis=0)
			total_in_branch = total_in_branch / overall_total
			branches = branches * total_in_branch
			return np.sum(branches)


		############################################################
		# TODO: compare each split using conditional entropy
		#       find the best split
		############################################################
		entropy = 9999
		for idx_dim in range(len(self.features[0])):
			s = list(set(map(lambda x: x[idx_dim], self.features)))
			if len(s) < 2:
				continue
			branches = [[0 for j in range(0, len(s))] for i in range(0, self.num_cls)]
			labels = list(set(self.labels))
			for i in range(len(self.features)):
				index = s.index(self.features[i][idx_dim])
				branches[labels.index(self.labels[i])][index] += 1
			curentropy = conditional_entropy(branches)
			if curentropy < entropy:
				entropy = curentropy
				self.dim_split = idx_dim
				self.feature_uniq_split = s

		if self.feature_uniq_split is None:
			self.splittable = False
			return

		############################################################
		# TODO: split the node, add child nodes
		###########################################################
		if self.splittable and len(self.feature_uniq_split) > 0:
			children = [[] for i in range(len(self.feature_uniq_split))]
			for i in range(len(self.features)):
				index = self.feature_uniq_split.index(self.features[i][self.dim_split])
				item = [self.features[i], self.labels[i]]
				children[index].append(item)
		else:
			children = list()

		final_children = list()
		for branch in children:
			labels = list()
			features = list()
			for each_child in branch:
				features.append(each_child[0])
				labels.append(each_child[1])
			num_cls = len(set(labels))
			features = np.array(features)
			features = np.delete(features, self.dim_split, axis=1)
			features = list(features)
			final_children.append(TreeNode(features, labels, num_cls))
		self.children = final_children

		# split the child nodes
		for child in self.children:
			if child.splittable:
				child.split()

		return

	def predict(self, feature: List[int]) -> int:
		if self.splittable:
			# print(feature)
			idx_child = self.feature_uniq_split.index(feature[self.dim_split])
			feature = feature[:self.dim_split] + feature[self.dim_split + 1:]
			return self.children[idx_child].predict(feature)
		else:
			return self.cls_max
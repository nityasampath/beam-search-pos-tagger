#!/usr/bin/env python3

import sys 
import math
from operator import attrgetter
from collections import defaultdict
#import heapq
#import math

#beamsearch_maxent.sh test_data boundary_file model_file sys_output beam_size topN topK
test_file = open(sys.argv[1], 'r')
boundary_file = open(sys.argv[2], 'r')
model_file = sys.argv[3]
sys_output = open(sys.argv[4], 'w')
beam_size = int(sys.argv[5])
topN = int(sys.argv[6])
topK = int(sys.argv[7])

model = {} #{class : {feature : weight}}
tags = []

class Node:
	def __init__(self, tag = None, prob = None, prevT = None, prev2T = None, parent = None, path_prob = None):
		self.tag = tag
		self.prob = prob
		self.prevT = prevT
		self.prev2T = prev2T
		self.parent = parent
		self.path_prob = path_prob

#read in model file
with open(model_file, 'r') as model_data:

	for line in model_data:
		line = line.strip()
		tokens = line.split(' ')

		if tokens[0] == 'FEATURES':
			tag = tokens[-1]
			tags.append(tag)
			model[tag] = {}
		else:
			feature = tokens[0]
			weight = float(tokens[1])
			model[tag][feature] = weight

#get test data
test_data = test_file.readlines()
test_file.close()


#get sentence lengths
sent_bounds = [int(line.strip()) for line in boundary_file]
boundary_file.close()

#set total word count and initialize correct words count for calculating accuracy
total_words = sum(sent_bounds)
correct_words = 0


sys_output.write('%%%%% test data:\n')

 #indexes for beginning and ending each beam search
sent_start = 0
sent_end = 0

#find tag set for each sentence
for sent_len in sent_bounds:

	beam_tree = [] #store tree nodes at each level

	sent_end += sent_len

	for i in range(sent_start, sent_end): #iterate through each word ina sentence

		word_vector = test_data[i].split() #get vector for current word

		#if beginning of sentence
		if i == sent_start:

			# calculate probs for each class
			Z = 0
			results = {}
			probs = {}

			for c in model.keys():
				sum_c = model[c]['<default>']

				for i in range(2, len(word_vector), 2):
					feature = word_vector[i]
					if feature in model[c]:
						sum_c += model[c][feature]

				if 'prevT=BOS' in model[c]:
					sum_c += model[c]['prevT=BOS']
				if 'prev2T=BOS+BOS' in model[c]:
					sum_c += model[c]['prev2T=BOS+BOS']


				results[c] = math.exp(sum_c)
				Z += results[c]

			for r in results:
				probs[r] = results[r]/float(Z)

			#get topN tags
			topN_tags_w1 = []
			
			for tag, prob in sorted(probs.items(), key=lambda item: (-item[1], item[0]))[:topN]:
				new_node = Node(tag, prob, 'prevT=BOS', 'prev2T=BOS+BOS', None, math.log10(prob))
				topN_tags_w1.append(new_node)

			beam_tree.append(topN_tags_w1)
		
		else: 

			topN_tags = []

			#iterate through nodes at each level
			for node in beam_tree[i-1-sent_start]:

				#set features for prevT and prev2T
				prevT = 'prevT=' + node.tag
				prev2T = 'prev2T=' + node.prevT.split('=')[1] + '+' + node.tag

				# calculate probs for each class
				Z = 0
				results = defaultdict(float)
				probs = {}

				for c in model.keys():
					sum_c = model[c]['<default>']

					for i in range(2, len(word_vector), 2):
						feature = word_vector[i]
						if feature in model[c]:
							sum_c += model[c][feature]

					if prevT in model[c]:	
						sum_c += model[c][prevT]
					if prev2T in model[c]:	
						sum_c += model[c][prev2T]

					results[c] = math.exp(sum_c)
					Z += results[c]

				for r in results:
					probs[r] = results[r]/float(Z)

				#get topN tags for current word and form new nodes
				for tag, prob in sorted(probs.items(), key=lambda item: (-item[1], item[0]))[:topN]:
					new_path_prob = node.path_prob + math.log10(prob)
					new_node = Node(tag, prob, prevT, prev2T, node, new_path_prob)
					topN_tags.append(new_node)

			#prune nodes at position i
			if topN_tags:
				max_prob = max(node.path_prob for node in topN_tags)

			topK_nodes = sorted(topN_tags, key=lambda node:node.path_prob, reverse = True)[:topK]

			pruned_topK = []
			for node in topK_nodes:
				if node.path_prob + beam_size >= max_prob:
					pruned_topK.append(node)

			beam_tree.append(pruned_topK)

	#pick node with highest probabiliy
	best_node = max(beam_tree[-1], key=attrgetter('path_prob'))
	
	#backtrack from best node to find path to root
	best_path = [best_node]
	curr_node = best_node
	while curr_node.parent is not None:
		best_path.append(curr_node.parent)
		curr_node = curr_node.parent

	best_path.reverse()

	#print words, tags and probabilities and count correct words
	for i in range(0, len(best_path)):
		test_line = test_data[i+sent_start]
		tokens = test_line.split()
		sys_output.write(str(tokens[0]) + ' ' + str(tokens[1]) + ' ' + str(best_path[i].tag) + ' ' + str(best_path[i].prob) + '\n')

		if tokens[1] == best_path[i].tag:
			correct_words += 1

	sent_start = sent_end #update start index

sys_output.close()

#calculate accuracy from count of correct words and print
accuracy = correct_words/total_words
print('Test accuracy=' + str(accuracy))











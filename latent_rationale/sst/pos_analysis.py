import os
import pickle
import numpy as np
import nltk

from latent_rationale.sst.vocabulary import Vocabulary
from latent_rationale.sst.models.model_helpers import build_model
from latent_rationale.sst.util import get_predict_args, sst_reader, \
	load_glove, print_parameters, get_device, find_ckpt_in_directory, \
	plot_dataset, get_scores
from latent_rationale.sst.evaluate import evaluate
from latent_rationale.sst.evaluate import get_prediction
from collections import namedtuple


Example = namedtuple("Example", ["tokens", "label", "token_labels"])

adjectives = {'JJ', 'JJR', 'JJS'}
nouns = {'NN', 'NNS'}
proper_nouns = {'NNP', 'NNPS'}
verbs = {'VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ'}
adverbs = {'RB', 'RBR', 'RBS'}
foreign_words = {'FW'}
pronouns = {'PRP', 'PRP$'}

def update_dict(pos, curr_pos):
	if curr_pos in adjectives:
		pos['adjective'] += 1
	elif curr_pos in nouns:
		pos['noun'] += 1
	elif curr_pos in proper_nouns:
		pos['proper noun'] += 1
	elif curr_pos in pronouns:
		pos['pronoun'] += 1
	elif curr_pos in verbs:
		pos['verb'] += 1
	elif curr_pos in adverbs:
		pos['adverb'] += 1
	elif curr_pos in foreign_words:
		pos['foreign word'] += 1
	else:
		pos['other'] += 1

def normalize_dict(d):
	total = 0
	for k in d:
		total += d[k]
	for k in d:
		d[k] /= total

def calculate_pos():
	predict_cfg = get_predict_args()
	ckpt_path = predict_cfg.ckpt
	with open(ckpt_path + '/data_results.pickle', 'rb') as handle:
		data_info = pickle.load(handle)

	total_pos = {'noun' : 0, 'proper noun' : 0, 'pronoun' : 0, 'verb' : 0, 'adjective' : 0, 'adverb' : 0, 'foreign word' : 0, 'other' : 0}
	original_selected = {'noun' : 0, 'proper noun' : 0, 'pronoun' : 0, 'verb' : 0, 'adjective' : 0, 'adverb' : 0, 'foreign word' : 0, 'other' : 0}
	perturbed_selected = {'noun' : 0, 'proper noun' : 0, 'pronoun' : 0, 'verb' : 0, 'adjective' : 0, 'adverb' : 0, 'foreign word' : 0, 'other' : 0}
	total_selected = {'noun' : 0, 'proper noun' : 0, 'pronoun' : 0, 'verb' : 0, 'adjective' : 0, 'adverb' : 0, 'foreign word' : 0, 'other' : 0}
	selection_changes_pos = {'noun' : 0, 'proper noun' : 0, 'pronoun' : 0, 'verb' : 0, 'adjective' : 0, 'adverb' : 0, 'foreign word' : 0, 'other' : 0}


	for i in data_info: # i is the number of the original
		original_sentence = data_info[i]['original'].tokens
		original_pred = data_info[i]['perturbed preds'][0]
		original_scores = data_info[i]['perturbed scores'][0]
		original_pos_tokens = nltk.pos_tag(original_sentence)

		for j, data in enumerate(data_info[i]['perturbed']): # j is the number of the perturbation for the current original
			perturbed_sentence = data_info[i]['perturbed'][j].tokens
			perturbed_pos_tokens = nltk.pos_tag(perturbed_sentence)
			scores = data_info[i]['perturbed scores'][j]
			if j == 0: # handle originals
				for k in range(len(scores)):
					curr_pos = original_pos_tokens[k][1]
					# get original selection distribution
					if scores[k] > 0:
						update_dict(original_selected, curr_pos)
						update_dict(total_selected, curr_pos)
			elif data_info[i]['perturbed preds'][j] == original_pred:
				for k in range(len(scores)): # k is the index in the perturbed sentence
					curr_pos = original_pos_tokens[k][1] # get pos tag
					pert_pos = perturbed_pos_tokens[k][1]
					# get perturbed selection distribution
					if scores[k] > 0:
						update_dict(perturbed_selected, pert_pos)
						update_dict(total_selected, pert_pos)
					# get selection change distribution
					if (scores[k] > 0) != (original_scores[k] > 0):
						update_dict(selection_changes_pos, curr_pos)
					# get background pos distribution
					update_dict(total_pos, curr_pos)

	normalize_dict(original_selected)
	normalize_dict(perturbed_selected)
	normalize_dict(total_selected)
	normalize_dict(selection_changes_pos)
	normalize_dict(total_pos)
	print('selection change:', selection_changes_pos)
	print()
	print('background (total):', total_pos)
	print()
	print('original selected: ', original_selected)
	print()
	print('perturbed selected: ', perturbed_selected)
	print()
	print('total selected: ', total_selected)

if __name__ == '__main__':
	calculate_pos()
import os
import pickle

import random
import numpy as np

from latent_rationale.sst.vocabulary import Vocabulary
from latent_rationale.sst.models.model_helpers import build_model
from latent_rationale.sst.util import get_predict_args, sst_reader, \
	load_glove, print_parameters, get_device, find_ckpt_in_directory, \
	plot_dataset, get_scores
from latent_rationale.sst.evaluate import evaluate
from latent_rationale.sst.evaluate import get_prediction
from collections import namedtuple

Example = namedtuple("Example", ["tokens", "label", "token_labels"])


def get_random_examples():
	total = 2210
	random.seed(0)
	np.random.seed(0)

	predict_cfg = get_predict_args()
	ckpt_path = predict_cfg.ckpt

	with open(ckpt_path + '/data_results.pickle', 'rb') as handle:
		data_info = pickle.load(handle)

	indices = random.sample([i for i in range(total)], 100)
	
	c = 0
	for i in indices:
		if len(data_info[i]['perturbed']) == 1: # no perturbation here
			continue
		for j in range(len(data_info[i]['perturbed'])):
			if j == 0: # skip original
				continue
			if data_info[i]['perturbed preds'][j] != data_info[i]['perturbed preds'][0]: # different pred
				continue
			if ((data_info[i]['perturbed scores'][0] > 0) != (data_info[i]['perturbed scores'][j] > 0)).sum() == 0: # no selection changes
				continue
			c += 1
			print('ex', c)
			print('orig: ' + ' '.join(data_info[i]['original'].tokens))
			print('pert: ' + ' '.join(data_info[i]['perturbed'][j].tokens))
			sel1 = []
			sel2 = []
			for k in range(len(data_info[i]['perturbed scores'][0])):
				if data_info[i]['perturbed scores'][0][k] > 0:
					sel1.append(data_info[i]['original'].tokens[k])
				if data_info[i]['perturbed scores'][j][k] > 0:
					sel2.append(data_info[i]['perturbed'][j].tokens[k])
			print('original selection:', sel1)
			print('perturbed selection:', sel2)
			print('original scores: ', data_info[i]['perturbed scores'][0])
			print('perturbed scores: ', data_info[i]['perturbed scores'][j])
			print()
			break

def get_change_examples(num_change):
	predict_cfg = get_predict_args()
	ckpt_path = predict_cfg.ckpt

	with open(ckpt_path + '/data_results.pickle', 'rb') as handle:
		data_info = pickle.load(handle)

	for i in data_info:
		for j in range(len(data_info[i]['perturbed'])):
			if j == 0: # skip original
				continue
			if data_info[i]['perturbed preds'][j] != data_info[i]['perturbed preds'][0]: # skip different pred
				continue
			if ((data_info[i]['perturbed scores'][0] > 0) != (data_info[i]['perturbed scores'][j] > 0)).sum() >= num_change:
				print('orig: ' + ' '.join(data_info[i]['original'].tokens))
				print('pert: ' + ' '.join(data_info[i]['perturbed'][j].tokens))
				sel1 = []
				sel2 = []
				for k in range(len(data_info[i]['perturbed scores'][0])):
					if data_info[i]['perturbed scores'][0][k] > 0:
						sel1.append(data_info[i]['original'].tokens[k])
					if data_info[i]['perturbed scores'][j][k] > 0:
						sel2.append(data_info[i]['perturbed'][j].tokens[k])
				print('original selection:', sel1)
				print('perturbed selection:', sel2)
				print('original scores: ', data_info[i]['perturbed scores'][0])
				print('perturbed scores: ', data_info[i]['perturbed scores'][j])


if __name__ == '__main__':
	pass
import os
import pickle
from collections import OrderedDict

import operator
import torch
import torch.optim
import numpy as np
import random
import nltk
from nltk.corpus import wordnet as wn
from transformers import BertTokenizer, BertModel, BertForMaskedLM

from latent_rationale.sst.vocabulary import Vocabulary
from latent_rationale.sst.models.model_helpers import build_model
from latent_rationale.sst.util import get_predict_args, sst_reader, \
	load_glove, print_parameters, get_device, find_ckpt_in_directory, \
	plot_dataset, get_scores
from latent_rationale.sst.evaluate import evaluate
from latent_rationale.sst.evaluate import get_prediction
from collections import namedtuple

Example = namedtuple("Example", ["tokens", "label", "token_labels"])

def predict():
	"""
	Make predictions with a saved model.
	"""

	predict_cfg = get_predict_args()
	device = get_device()
	print(device)

	# load checkpoint
	ckpt_path = find_ckpt_in_directory(predict_cfg.ckpt)
	ckpt = torch.load(ckpt_path, map_location=device)
	cfg = ckpt["cfg"]

	for k, v in cfg.items():
		print("{:20} : {:10}".format(k, v))

	batch_size = 1 # evaluate individually

	# Let's load the data into memory.
	with open('data/sst/data_info.pickle', 'rb') as handle:
		perturbed_test_data_info = pickle.load(handle)
	
	example = perturbed_test_data_info[0]
	print("First perturbed test example:", example)


	vocab = Vocabulary()
	vectors = load_glove(cfg["word_vectors"], vocab)  # this populates vocab

	# Map the sentiment labels 0-4 to a more readable form (and the opposite)
	i2t = ["very negative", "negative", "neutral", "positive", "very positive"]
	t2i = OrderedDict({p: i for p, i in zip(i2t, range(len(i2t)))})

	# Build model
	model = build_model(cfg["model"], vocab, t2i, cfg)

	# load parameters from checkpoint into model
	print("Loading saved model..")
	model.load_state_dict(ckpt["state_dict"])
	model.to(device)
	model.eval()
	print("Done")

	# print model
	print(model)
	print_parameters(model)

	print("Evaluating")

	words_selected = 0
	words_total = 0
	correct = 0
	total = 0
	classes = [0,0,0,0,0]
	selection_change_frequencies = {}
	for i in perturbed_test_data_info:
		perturbed_test_data_info[i]['perturbed preds'] = []
		perturbed_test_data_info[i]['perturbed scores'] = []
		original_sentence = [perturbed_test_data_info[i]['original']]
		original_scores = get_scores(model, original_sentence, device=None).numpy().flatten()
		original_pred = get_prediction(model, original_sentence, device=None).item()

		words_selected += (original_scores > 0).sum()
		words_total += len(original_scores)
		for j,perturbed_input in enumerate(perturbed_test_data_info[i]['perturbed']):
			if j == 0: # skip originals
				continue 
			curr_input = [perturbed_input]
			pred = get_prediction(model, curr_input, device=None).item()
			classes[pred] += 1
			scores = get_scores(model, curr_input, device=None)
			scores = scores.numpy().flatten()
			selection_changes = ((original_scores > 0) != (scores > 0)).sum()

			if pred == original_pred:
				if selection_changes not in selection_change_frequencies:
					selection_change_frequencies[selection_changes] = 1
				else:
					selection_change_frequencies[selection_changes] += 1

			perturbed_test_data_info[i]['perturbed preds'].append(pred)
			perturbed_test_data_info[i]['perturbed scores'].append(scores)
			if pred == perturbed_input.label:
				correct += 1
			total += 1

	print('selection change frequencies:', selection_change_frequencies)
	print('class frequencies:', classes)
	with open(predict_cfg.ckpt + 'data_results.pickle', 'wb') as handle:
		pickle.dump(perturbed_test_data_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

	acc = correct / total
	selection_fraction = words_selected / words_total
	print('correct:', correct)
	print('total:', total)
	print('accuracy:', acc)
	print('selection fraction:', selection_fraction)

if __name__ == "__main__":
	predict()
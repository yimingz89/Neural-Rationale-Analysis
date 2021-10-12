import os
import pickle

import random
import numpy as np
import nltk
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
from collections import defaultdict
from scipy.stats import gaussian_kde

from latent_rationale.sst.vocabulary import Vocabulary
from latent_rationale.sst.models.model_helpers import build_model
from latent_rationale.sst.util import get_predict_args, sst_reader, \
	load_glove, print_parameters, get_device, find_ckpt_in_directory, \
	plot_dataset, get_scores
from latent_rationale.sst.evaluate import evaluate
from latent_rationale.sst.evaluate import get_prediction
from collections import namedtuple

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

WIDTH = 4.5
HEIGHT = 3.375

Example = namedtuple("Example", ["tokens", "label", "token_labels"])

def get_stability_data(data_info):
	stability_ratio = []
	change_perturbations = []
	num_perturbations = []
	sentence_lengths = []
	for i in data_info: # i is the number of the original
		original_pred = data_info[i]['perturbed preds'][0]
		original_scores = data_info[i]['perturbed scores'][0]
		total_perturbs_same_pred = 0
		perturbs_same_pred_selection_change = 0
		for j, data in enumerate(data_info[i]['perturbed']): # j is the number of the perturbation for the current original
			if j == 0: # skip originals
				continue
			if data_info[i]['perturbed preds'][j] == original_pred:
				total_perturbs_same_pred += 1
				scores = data_info[i]['perturbed scores'][j]
				selection_changes = ((original_scores > 0) != (scores > 0)).sum()
				if selection_changes > 0:
					perturbs_same_pred_selection_change += 1
		if total_perturbs_same_pred == 0: # if no perturbations with same pred, continue
			continue
		curr_ratio = perturbs_same_pred_selection_change / total_perturbs_same_pred
		stability_ratio.append(curr_ratio)
		sentence_lengths.append(len(original_scores))
		change_perturbations.append(perturbs_same_pred_selection_change)
		num_perturbations.append(total_perturbs_same_pred)

	data_map = {} # map of # perturbation -> # change) -> frequency
	for i in range(len(num_perturbations)):
		curr_data = (num_perturbations[i], change_perturbations[i])
		if curr_data[0] not in data_map:
			data_map[curr_data[0]] = {}
			data_map[curr_data[0]]['max'] = 0
		if curr_data[1] not in data_map[curr_data[0]]:
			data_map[curr_data[0]][curr_data[1]] = 0
		data_map[curr_data[0]][curr_data[1]] += 1
		data_map[curr_data[0]]['max'] = max(data_map[curr_data[0]]['max'], data_map[curr_data[0]][curr_data[1]])
	return data_map

def get_indices_data(data_info):
	indices_of_selection_changes = [] # only consider perturbed inputs with same prediction as original
	sentence_lengths = []
	for i in data_info: # i is the number of the original
		original_pred = data_info[i]['perturbed preds'][0]
		original_scores = data_info[i]['perturbed scores'][0]
		for j, data in enumerate(data_info[i]['perturbed']): # j is the number of the perturbation for the current original
			if j == 0: # skip originals
				continue
			if data_info[i]['perturbed preds'][j] == original_pred:
				scores = data_info[i]['perturbed scores'][j]
				for k in range(len(scores)): # k is the index in the perturbed sentence
					if (scores[k] > 0) != (original_scores[k] > 0):
						indices_of_selection_changes.append(k)
						sentence_lengths.append(len(scores))

	return (sentence_lengths, indices_of_selection_changes)


def get_distance_data(data_info):
	dist_to_change_frequencies_by_length = {} # only consider perturbed inputs with same prediction as original
	for i in data_info:
		original_sentence = data_info[i]['original'].tokens
		original_pred = data_info[i]['perturbed preds'][0]
		original_scores = data_info[i]['perturbed scores'][0]
		length = len(original_scores)
		if length not in dist_to_change_frequencies_by_length:
			dist_to_change_frequencies_by_length[length] = {'zero': 0, 'non-zero' : []}
		for j, data in enumerate(data_info[i]['perturbed']):
			if j == 0: # skip originals
				continue 
			if data_info[i]['perturbed preds'][j] == original_pred:
				scores = data_info[i]['perturbed scores'][j]
				selection_changes = ((original_scores > 0) != (scores > 0))
				perturbed_sentence = data_info[i]['perturbed'][j].tokens
				index_of_change = -1
				for k in range(len(original_sentence)):
					if original_sentence[k] != perturbed_sentence[k]:
						index_of_change = k
				if index_of_change == -1:
					raise ValueError('The index of change should not be -1')
				for k in range(length):
					if selection_changes[k]: # if the selection changes, find the distance to perturbed word
						dist = abs(k - index_of_change)
						if dist == 0:
							dist_to_change_frequencies_by_length[length]['zero'] += 1
						else:
							dist_to_change_frequencies_by_length[length]['non-zero'].append(dist)
	total_zero = 0
	total_nonzero = 0
	for l in dist_to_change_frequencies_by_length:
		total_zero += dist_to_change_frequencies_by_length[l]['zero']
		total_nonzero += len(dist_to_change_frequencies_by_length[l]['non-zero'])

	length_by_fives = {}
	for length in dist_to_change_frequencies_by_length:
		div = int(length / 5)
		if div not in length_by_fives:
			length_by_fives[div] = []
		length_by_fives[div].extend(dist_to_change_frequencies_by_length[length]['non-zero'])

	mins = []
	maxs = []
	first_quartiles = []
	third_quartiles = []
	meds = []
	for length in length_by_fives:
		curr_dists = length_by_fives[length]
		if len(curr_dists) == 0:
			continue
		x_pos = length*5 + 2
		mins.append((x_pos, min(curr_dists)))
		first_quartiles.append((x_pos, np.quantile(curr_dists, 0.25)))
		meds.append((x_pos, np.quantile(curr_dists, 0.5)))
		third_quartiles.append((x_pos, np.quantile(curr_dists, 0.75)))
		maxs.append((x_pos, max(curr_dists)))

	mins = sorted(mins, key=lambda x: x[0])
	first_quartiles = sorted(first_quartiles, key=lambda x: x[0])
	meds = sorted(meds, key=lambda x: x[0])
	third_quartiles = sorted(third_quartiles, key=lambda x: x[0])
	maxs = sorted(maxs, key=lambda x: x[0])

	return ([i[0] for i in first_quartiles], [i[1] for i in first_quartiles], [i[0] for i in meds], [i[1] for i in meds], [i[0] for i in third_quartiles], [i[1] for i in third_quartiles])


def plot_all():
	fig = plt.figure(figsize=[10, 2.5])
	gs = gridspec.GridSpec(ncols=9, nrows=1, figure=fig, wspace=0, 
						   width_ratios=[10, 1, 10, 3.5, 10, 1, 10, 3, 10])

	plt.subplot(gs[0, 0])
	with open('results/sst/latent_30pct/data_results.pickle', 'rb') as handle:
		data_info_latent = pickle.load(handle)
	with open('results/sst/bernoulli_sparsity01505/data_results.pickle', 'rb') as handle:
		data_info_bernoulli = pickle.load(handle)

	x_latent_first, y_latent_first, x_latent_med, y_latent_med, x_latent_third, y_latent_third = get_distance_data(data_info_latent)
	x_bernoulli_first, y_bernoulli_first, x_bernoulli_med, y_bernoulli_med, x_bernoulli_third, y_bernoulli_third = get_distance_data(data_info_bernoulli)

	# plot CR data
	x_latent_first[-1] = 52
	x_latent_med[-1] = 52
	x_latent_third[-1] = 52
	plt.plot(x_latent_first[1:], y_latent_first[1:], '.-')
	plt.plot(x_latent_med[1:], y_latent_med[1:], '.-')
	plt.plot(x_latent_third[1:], y_latent_third[1:], '.-')

	plt.gca().set_xlabel('sent. len.')
	plt.gca().set_ylabel('dist. to pertb.', labelpad=0)
	plt.text(7, 6.5, 'CR model', ha='left', va='center')

	plt.yticks(np.arange(0,8,2))
	labels = ['5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50+']
	plt.xticks(x_latent_first[1:], labels, rotation='vertical')
	ylim = plt.gca().get_ylim()

	plt.subplot(gs[0, 2])
	# plot PG data
	x_bernoulli_first[-1] = 52
	x_bernoulli_med[-1] = 52
	x_bernoulli_third[-1] = 52
	plt.plot(x_bernoulli_first, y_bernoulli_first, '.-')
	plt.plot(x_bernoulli_med, y_bernoulli_med, '.-')
	plt.plot(x_bernoulli_third, y_bernoulli_third, '.-')

	plt.gca().set_xlabel('sent. len.')
	labels = ['5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50+']
	plt.xticks(x_bernoulli_first, labels, rotation='vertical')
	plt.yticks([])
	plt.text(7, 6.5, 'PG model', ha='left', va='center')
	plt.gca().set_ylim(*ylim)

	plt.subplot(gs[0, 4])
	with open('results/sst/latent_30pct/data_results.pickle', 'rb') as handle:
		data_info_latent = pickle.load(handle)
	with open('results/sst/bernoulli_sparsity01505/data_results.pickle', 'rb') as handle:
		data_info_bernoulli = pickle.load(handle)

	x_latent, y_latent = get_indices_data(data_info_latent)
	x_bernoulli, y_bernoulli = get_indices_data(data_info_bernoulli)
	marker_size = matplotlib.rcParams['lines.markersize'] ** 2.

	# plot CR data
	plt.scatter(x_latent, y_latent, alpha=0.2, edgecolors='none', s=marker_size/7)
	plt.gca().set_xlabel('sent. len.', labelpad=0)
	plt.xticks([10, 30, 50])
	plt.gca().set_ylabel('change loc.', labelpad=0)
	ylim = plt.gca().get_ylim()
	plt.text(4, 45.5, 'CR model', ha='left', va='center')

	plt.subplot(gs[0, 6])
	# plot PG data
	plt.scatter(x_bernoulli, y_bernoulli, alpha=0.2, edgecolors='none', s=marker_size/7)
	plt.gca().set_xlabel('sent. len.', labelpad=0)
	plt.yticks([])
	plt.xticks([10, 30, 50])
	plt.gca().set_ylim(*ylim)
	plt.text(4, 45.5, 'PG model', ha='left', va='center')

	plt.subplot(gs[0, 8])
	with open('results/sst/latent_30pct/data_results.pickle', 'rb') as handle:
		data_info_latent = pickle.load(handle)
	with open('results/sst/bernoulli_sparsity01505/data_results.pickle', 'rb') as handle:
		data_info_bernoulli = pickle.load(handle)

	data_map_latent = get_stability_data(data_info_latent)
	data_map_bernoulli = get_stability_data(data_info_bernoulli)
	
	ax = plt.gca()

	# plot latent stabilities
	for i in data_map_latent:
		max_freq = data_map_latent[i]['max']
		if i in data_map_bernoulli:
			max_freq = max(max_freq, data_map_bernoulli[i]['max'])
		for j in data_map_latent[i]:
			if j == 'max': # skip auxiliary data
				continue
			curr_freq = data_map_latent[i][j]
			lower_left = (i, j+0.05)
			length = 0.8 * (curr_freq / max_freq)
			height = 0.2
			ax.add_patch(Rectangle(lower_left, length, height, color='C0'))
	# plot bernoulli stabilities
	for i in data_map_bernoulli:
		max_freq = data_map_bernoulli[i]['max']
		if i in data_map_latent:
			max_freq = max(max_freq, data_map_latent[i]['max'])
		for j in data_map_bernoulli[i]:
			if j == 'max': # skip auxiliary data
				continue
			curr_freq = data_map_bernoulli[i][j]
			lower_left = (i, j - 0.25)
			length = 0.8 * (curr_freq / max_freq)
			height = 0.2
			ax.add_patch(Rectangle(lower_left, length, height, color='C1'))
	plt.xlabel('# pertb.', labelpad=0)
	plt.ylabel('# change', labelpad=0)
	plt.xlim([1,8])
	plt.ylim([-0.5,6.5])
	for i in range(7):
		plt.axhline(y=i, color='C4', linestyle='--', linewidth = 0.5)
	for i in range(1,10,1):
		plt.axvline(x=i, color = 'black', linestyle='-.', linewidth = 0.5)
	plt.xticks([1, 3, 5, 7, 9])
	plt.yticks([0, 2, 4, 6])
	custom_lines = [Patch(facecolor='C0', edgecolor='none'),
                	Patch(facecolor='C1', edgecolor='none')]
	plt.legend(custom_lines, ['CR model', 'PG model'], loc='upper left', 
		prop={'size': 7})

	plt.tight_layout()
	plt.savefig('paper_figures/all1.pdf', bbox_inches='tight')
	plt.show()

if __name__ == '__main__':
	plot_all()
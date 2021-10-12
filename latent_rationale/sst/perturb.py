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
    plot_dataset
from latent_rationale.sst.evaluate import evaluate
from collections import namedtuple
 
def perturb():
    data_info = {}
    test_data = list(sst_reader("data/sst/test.txt"))
    print('perturbing...')
    perturbed_test_data, data_info = perturb_bert_wordnet_intersection(test_data)
    with open('data/sst/data_info.pickle', 'wb') as handle:
        pickle.dump(data_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('done perturbing')


Example = namedtuple("Example", ["tokens", "label", "token_labels"])

# perturb data, replacing words (specifically forms of adjectives, nouns, or verbs) with a replacement from the intersection of (30) bert predictions and wordnet synonym set
def perturb_bert_wordnet_intersection(data):
    perturbed_dict = {}
    changed_count = 0
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    model.eval()
    res = []
    for num, d in enumerate(data):
        perturbed_dict[num] = {}
        perturbed_dict[num]['original'] = d
        perturbed_dict[num]['perturbed'] = []
        perturbed_tokens_list = replace_random_single(d.tokens, tokenizer, model)
        # append both unchanged and changed data (we can filter out unchanged later in the analysis)
        for perturbed_tokens in perturbed_tokens_list:
            # note that the first perturbed_token will be the original tokens (which is fine)
            res.append(Example(perturbed_tokens, label=d.label, token_labels=d.token_labels))
            perturbed_dict[num]['perturbed'].append(Example(perturbed_tokens, label=d.label, token_labels=d.token_labels))
        changed_count += (len(perturbed_tokens_list) - 1)
    return res, perturbed_dict

# get all valid perturbations of the input tokens (single word replacement) found using the intersection of BERT and wordnet of
def replace_random_single(tokens, tokenizer, model):
    sentence = ' '.join(tokens)
    tags = nltk.pos_tag(nltk.word_tokenize(sentence))
    adjs = []
    for i, tag in enumerate(tags):
        # get all indices with some form of an adjectice, noun, or verb
        if tag[1] in {'JJ', 'NN', 'NNS', 'VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ'}:
            adjs.append(i)
    perturbations = [tokens]

    # loop through potential indices, get all valid perturbations (i.e. where change = True)
    for i in adjs:
        if i < len(tokens):
            new_tokens, change = replace_word_bert_wordnet_intersection(tokens, i, tokenizer, model)
            if change:
                perturbations.append(new_tokens)

    return perturbations


# replace a word using the bert, wordnet synonym net intersection and returns (tokens, change) where change is a boolean if the tokens were changed or not
def replace_word_bert_wordnet_intersection(tokens, index, tokenizer, model):
    changed = False
    text = " ".join(tokens)
    padded_text = '[CLS] ' + text + ' [SEP]'
    tokenized_text = tokenizer.tokenize(padded_text)
    word_to_replace = tokens[index]

    masked_index = -1

    num_occurrences = tokens[:index+1].count(tokens[index])

    # find occurence in bert tokenized text
    occurrence = 0
    for i in range(len(tokenized_text)):
        if tokenized_text[i] == word_to_replace:
            occurrence += 1
            if occurrence == num_occurrences:
                masked_index = i
    if masked_index == -1: # word somehow got split by the BERT tokenizer -> throw away example
        return (tokens, changed)

    tokenized_text[masked_index] = '[MASK]'
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])

    synonyms = []
    for synset in wn.synsets(word_to_replace):
        for lemma in synset.lemmas():
            synonyms.append(lemma.name())

    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]
    predicted_indices = torch.topk(predictions[0, masked_index], k=30).indices
    replacement_word = word_to_replace
    for i in range(len(predicted_indices)):
        curr_pred = tokenizer.convert_ids_to_tokens([predicted_indices[i].item()])[0]
        if curr_pred != word_to_replace and curr_pred.lower() in synonyms:
            replacement_word = curr_pred
            changed = True
            break

    new_tokens = tokens.copy()
    new_tokens[index] = replacement_word
    
    return (new_tokens, changed)

if __name__ == "__main__":
    perturb()
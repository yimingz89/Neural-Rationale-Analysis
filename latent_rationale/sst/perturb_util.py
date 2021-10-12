import os

import operator
import numpy as np
import random
import math

from latent_rationale.sst.vocabulary import Vocabulary
from collections import namedtuple

Example = namedtuple("Example", ["tokens", "label", "token_labels"])

def perturb_data(data, vectors, vocab):
    res = []
    for num, d in enumerate(data):
        print(num)
        perturbed_tokens = replace_with_random_synonyms(d.tokens, vectors, vocab)
        res.append(Example(perturbed_tokens, label=d.label, token_labels=d.token_labels))
        print(res)
    return res

def find_closest_synonym(word, vectors, vocab):
    if word not in vocab.w2i:
        return (word, 0) # if the word is not in our vocabulary, just keep it
    word_vec = vectors[vocab.w2i[word]]
    best_dist = math.inf # arbitrarily high starting distance
    best_word = None
    for i in range(len(vectors)):
        if i == vocab.w2i[word]: # skip over identity
            continue
        curr_vec = vectors[i]
        curr_dist = np.linalg.norm(word_vec - curr_vec)
        if curr_dist < best_dist:
            best_dist = curr_dist
            best_word = vocab.i2w[i]
    return (best_word, best_dist)

# k = max num of tokens to replace
def replace_with_top_synonyms(tokens, vectors, vocab, k=5):
    k = min(len(tokens), k)
    synonyms = []
    for i in range(len(tokens)):
        if tokens[i] not in vocab.w2i: # skip over tokens unknown to our vocab
            continue
        synonym, dist = find_closest_synonym(tokens[i], vectors, vocab)
        synonyms.append((i, synonym, dist))
    synonyms.sort(key = operator.itemgetter(2)) # sort by best_dist
    for i in range(len(synonyms)):
        tokens[synonyms[i][0]] = synonyms[i][1] # replace tokens with synonyms with k lowest distances

# replace_with_top_synonyms randomized, since computing everything is too slow
def replace_with_random_synonyms(tokens, vectors, vocab, k=5):
    k = min(len(tokens), k)
    to_replace = set(random.sample(range(0, len(tokens)), k)) # sample without replacement k random indices
    replaced = []
    for i in range(len(tokens)):
        if i in to_replace:
            synonym, dist = find_closest_synonym(tokens[i], vectors, vocab)
            replaced.append(synonym) # replace token with closest synonym
        else:
            replaced.append(tokens[i])
    return replaced
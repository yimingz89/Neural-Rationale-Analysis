import os
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

    batch_size = cfg.get("eval_batch_size", 25)

    # Let's load the data into memory.
    train_data = list(sst_reader("data/sst/train.txt"))
    dev_data = list(sst_reader("data/sst/dev.txt"))
    test_data = list(sst_reader("data/sst/test.txt"))

    print("train", len(train_data))
    print("dev", len(dev_data))
    print("test", len(test_data))

    example = test_data[0]
    print("First train example:", example)
    print("First train example tokens:", example.tokens)
    print("First train example label:", example.label)


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

    test_eval = evaluate(model, test_data, batch_size=batch_size, device=device)
    print("test acc", test_eval["acc"])

    print("Plotting attention scores")
    if predict_cfg.plot:
        plot_save_path = os.path.join(cfg["save_path"], "plots")
        if not os.path.exists(plot_save_path):
            os.makedirs(plot_save_path)
        plot_dataset(model, test_data, batch_size=batch_size,
                     device=device, save_path=plot_save_path)

Example = namedtuple("Example", ["tokens", "label", "token_labels"])

# perturb data, replacing words (specifically forms of adjectives, nouns, or verbs) with a replacement from the intersection of (30) bert predictions and wordnet synonym set
def perturb_bert_wordnet_intersection(data):
    changed_count = 0
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    model.eval()
    res = []
    for num, d in enumerate(data):
        perturbed_tokens, change = replace_random_single(d.tokens, tokenizer, model)
        if change:
            res.append(Example(perturbed_tokens, label=d.label, token_labels=d.token_labels))
            changed_count += 1
        else:
            res.append(Example(perturbed_tokens, label=d.label, token_labels=d.token_labels))
    return res

# replace up to 10 words with synonyms (found using the intersection of BERT and wordnet)
def replace_random_single(tokens, tokenizer, model):
    sentence = ' '.join(tokens)
    tags = nltk.pos_tag(nltk.word_tokenize(sentence))
    adjs = []
    for i, tag in enumerate(tags):
        # get all indices with some form of an adjectice, noun, or verb
        if tag[1] in {'JJ', 'NN', 'NNS', 'VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ'}:
            adjs.append(i)
    to_replace = []

    # loop through potential indices, getting rid of words which we cannot find a replacement for 
    for i in adjs:
        if i < len(tokens):
            new_tokens, change = replace_word_bert_wordnet_intersection(tokens, i, tokenizer, model)
            if change:
                to_replace.append(i)

    # sample random index to replace (from those which give changes)
    if len(to_replace) > 0: 
        replace_index = random.sample(to_replace, 1)[0]
        new_tokens, change = replace_word_bert_wordnet_intersection(tokens, replace_index, tokenizer, model)
        return new_tokens, change
    else:
        return tokens, False


# replace a word using the bert, wordnet synonym net intersection, returns (tokens, change) where change is a boolean if the tokens were changed or not
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
    predict()

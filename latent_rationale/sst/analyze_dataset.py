from collections import OrderedDict
from latent_rationale.sst.util import sst_reader
import math

# script for analyzing the dataset

train_data = list(sst_reader("data/sst/train.txt"))
dev_data = list(sst_reader("data/sst/dev.txt"))
test_data = list(sst_reader("data/sst/test.txt"))

print("Number of data by split:\n")
print("train length:", len(train_data))
print("dev length:", len(dev_data))
print("test length", len(test_data))
print("\n")

train_labels = {}
dev_labels = {}
test_labels = {}
sum_train_len, sum_dev_len, sum_test_len = 0, 0, 0
min_train_len, min_dev_len, min_test_len = math.inf, math.inf, math.inf
max_train_len, max_dev_len, max_test_len = 0, 0, 0

# compute label statistics and input length statistics
for d in train_data:
	curr_label = d.label
	if curr_label not in train_labels:
		train_labels[curr_label] = 1
	else:
		train_labels[curr_label] += 1
	sum_train_len += len(d.tokens)
	min_train_len = min(min_train_len, len(d.tokens))
	max_train_len = max(max_train_len, len(d.tokens))
for d in dev_data:
	curr_label = d.label
	if curr_label not in dev_labels:
		dev_labels[curr_label] = 1
	else:
		dev_labels[curr_label] += 1
	sum_dev_len += len(d.tokens)
	min_dev_len = min(min_dev_len, len(d.tokens))
	max_dev_len = max(max_dev_len, len(d.tokens))
for d in test_data:
	curr_label = d.label
	if curr_label not in test_labels:
		test_labels[curr_label] = 1
	else:
		test_labels[curr_label] += 1
	sum_test_len += len(d.tokens)
	min_test_len = min(min_test_len, len(d.tokens))
	max_test_len = max(max_test_len, len(d.tokens))

print("Label frequencies:\n")
print("train label frequencies:", train_labels)
print("dev label frequencies:", dev_labels)
print("test label frequencies", test_labels)

print("\n\nInput lengths by split:\n")
print("train input max len:", max_train_len)
print("train input min len:", min_train_len)
print("train input avg len:", sum_train_len/len(train_data))
print("")

print("dev input max len:", max_dev_len)
print("dev input min len:", min_dev_len)
print("dev input avg len:", sum_dev_len/len(dev_data))
print("")

print("test input max len:", max_test_len)
print("test input min len:", min_test_len)
print("test input avg len:", sum_test_len/len(test_data))
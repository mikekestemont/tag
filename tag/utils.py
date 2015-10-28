#!usr/bin/env python

import ConfigParser
import os
import glob
import logging
from collections import Counter
from operator import itemgetter

import numpy as np
#import editdist
from keras.utils import np_utils

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

def parse_cmd_line(config_path):
    """
    Parses the configuration file.
    Input: path to config-file
    Returns: a parameter dict
    """

    config = ConfigParser.ConfigParser()
    config.read(config_path)

    param_dict = dict()

    for section in config.sections():
        for name, value in config.items(section):

            if value.isdigit():
                value = int(value)
            elif value == "True":
                value = True
            elif value == "False":
                value = False

            param_dict[name] = value

    return param_dict

def load_data_file(filepath, nb_instances=5000, label_idxs=[1]):
    tokens, labels = [], []
    with open(filepath, 'r') as data_file:
        for line in data_file:
            line = line.strip()
            if not line:
                token, label = "$", "$" # mark beginning of utterances
            elif line.startswith("@"):
                token, label = "@", "@" # mark beginning of documents
            else:
                comps = line.strip().split("\t")
                token = comps[0]
                try:
                    label = '_'.join(comps[i].replace(' ', '').strip() for i in label_idxs)
                except:
                    continue
                token = token.replace('~', '')
                token = token.replace(' ', '')
                tokens.append(token)
                labels.append(label)
            if len(tokens) >= nb_instances:
                return tokens, labels
    return tokens, labels

def load_data_dir(data_dir, nb_instances=10000, label_idxs=[1]):
    tokens, labels = [], []
    if os.path.isdir(data_dir):
        for filepath in glob.glob(data_dir+"/*"):
            ts, ls = load_data_file(filepath, nb_instances, label_idxs)
            tokens.extend(ts)
            labels.extend(ls)
    return tuple(tokens), tuple(labels)

def get_char_vector_dict(tokens):
    char_vocab = tuple({ch for tok in tokens+('%', '$') for ch in tok+" "})
    char_vector_dict = {}
    filler = np.zeros(len(char_vocab), dtype="int8")
    for idx, char in enumerate(char_vocab):
        ph = filler.copy()
        ph[idx] = 1
        char_vector_dict[char] = ph
    return char_vector_dict

def get_train_token_index(tokens):
    counter = Counter(tokens)
    index = {'<unk>':0}
    for k, v in counter.items():
        if v >= 1:
            index[k] = len(index)
    return index

def vectorize_charseq(seq, char_vector_dict, std_seq_len):
    seq_X = []
    filler = np.zeros(len(char_vector_dict), dtype="int8")
    # cut, if needed:
    seq = seq[:std_seq_len]
    for char in seq:
        try:
            seq_X.append(char_vector_dict[char])
        except KeyError:
            seq_X.append(filler)
    # pad, if needed:
    while len(seq_X) < std_seq_len:
        seq_X.append(filler)
    return np.vstack(seq_X)

def accuracies(predictions, gold_labels, test_tokens, train_token_set):
    # first overall:
    all_predictions = np_utils.categorical_probas_to_classes(predictions['label_output'])
    all_acc = np_utils.accuracy(all_predictions, gold_labels)

    # split out known and unknown:
    known_predictions, unknown_predictions = [], []
    known_gold_labels, unknown_gold_labels = [], []
    for test_token, prediction, gold_label in zip(test_tokens, all_predictions, gold_labels):
        if test_token in train_token_set:
            known_gold_labels.append(gold_label)
            known_predictions.append(prediction)
        else:
            unknown_gold_labels.append(gold_label)
            unknown_predictions.append(prediction)

    # get known and unknown accuracies:
    known_acc = np_utils.accuracy(known_predictions, known_gold_labels)
    unknown_acc = np_utils.accuracy(unknown_predictions, unknown_gold_labels)
    return all_acc, known_acc, unknown_acc

def baseline(train_tokens, train_labels, test_tokens, test_labels):
    train_tokens = [t for t in train_tokens if t not in ("@", "$")]
    test_tokens = [t for t in test_tokens if t not in ("@", "$")]
    train_dict = {}

    for token, label in zip(train_tokens, train_labels):
        if token not in train_dict:
            train_dict[token] = {}
        if label not in train_dict[token]:
            train_dict[token][label] = 0
        train_dict[token][label] += 1

    nb_known_test_tokens, nb_unknown_test_tokens = 0.0, 0.0
    correct_known_test_tokens, correct_unknown_test_tokens = 0.0, 0.0

    for test_token, test_label in zip(test_tokens, test_labels):
        known_token = False

        # determine nearest neighbor via Levenshtein:
        if test_token in train_dict: # shortcut if the target is verbatimly present in train
            nn = test_token
            known_token = True
            nb_known_test_tokens += 1
        else:
            nb_unknown_test_tokens += 1
            candidates = train_dict.keys()
            distances = [(editdist.distance(test_token, c), c) for c in candidates]
            nn = min(distances, key=itemgetter(0))[1]

        # select most frequent label:
        proposed_label = max(train_dict[nn].iteritems(), key=itemgetter(1))[0]

        # check whether correct
        if proposed_label == test_label:
            if known_token:
                correct_known_test_tokens += 1
            else:
                correct_unknown_test_tokens += 1
    
    known_acc = correct_known_test_tokens/float(nb_known_test_tokens)
    unknown_acc = correct_unknown_test_tokens/float(nb_known_test_tokens)
    all_acc = (correct_known_test_tokens+correct_unknown_test_tokens)/float(nb_known_test_tokens+nb_unknown_test_tokens)
    proportion_unknown = nb_unknown_test_tokens/float(nb_known_test_tokens+nb_unknown_test_tokens)

    return all_acc, known_acc, unknown_acc, proportion_unknown


def vectorize(tokens, std_token_len, nb_left_tokens, left_char_len,
              nb_right_tokens, right_char_len, target_representation,
              include_target_one_hot, context_representation,
              char_vector_dict={}, train_token_index=None,
              ):

    left_X, tokens_X, right_X, target_one_hots = [], [], [], []

    filler = np.zeros(len(train_token_index))

    for token_idx, token in enumerate(tokens):

        # ignore boundary markers:
        if token in ("@", "$"):
            continue

        if include_target_one_hot:
            """
            f = filler
            try:
                f[train_token_index[token]] = 1.0
            except KeyError:
                pass
            target_one_hots.append(f)
            """
            try:
                target_one_hots.append([train_token_index[token]])
            except KeyError:
                target_one_hots.append([0])

        # vectorize target token:
        if target_representation in ('flat_characters', 'lstm_characters', 'flat_convolution', 'lstm_convolution'):
            tokens_X.append(vectorize_charseq(token, char_vector_dict, std_token_len))
        elif target_representation == 'embedding':
            try:
                tokens_X.append([train_token_index[token]])
            except KeyError:
                tokens_X.append([0])

        if context_representation in ('flat_characters', 'lstm_characters', 'flat_convolution', 'lstm_convolution'):
            # vectorize left context:
            left_context = [tokens[token_idx-(t+1)] for t in range(nb_left_tokens)
                                 if token_idx-(t+1) >= 0][::-1]
            left_str = " ".join(left_context)
            left_X.append(vectorize_charseq(left_str, char_vector_dict, left_char_len))

            # vectorize right context:
            right_str = " ".join([tokens[token_idx+t+1] for t in range(nb_right_tokens)
                                 if token_idx+t+1 < len(tokens)])
            right_X.append(vectorize_charseq(right_str, char_vector_dict, right_char_len))

        elif context_representation in ('flat_embeddings', 'lstm_embeddings'):
            # vectorize left context:
            left_context_tokens = [tokens[token_idx-(t+1)] for t in range(nb_left_tokens)
                                 if token_idx-(t+1) >= 0][::-1]
            idxs = []
            if left_context_tokens:
                idxs = [train_token_index[t] if t in train_token_index else 0 for t in left_context_tokens]
            while len(idxs) < nb_left_tokens:
                idxs = [0] + idxs
            left_X.append(idxs)

            # vectorize right context:
            right_context_tokens = [tokens[token_idx+t+1] for t in range(nb_right_tokens) if token_idx+t+1 < len(tokens)]
            idxs = []
            if right_context_tokens:
                idxs = [train_token_index[t] if t in train_token_index else 0 for t in right_context_tokens]
            while len(idxs) < nb_right_tokens:
                idxs.append(0)
            right_X.append(idxs)

    tokens_X = np.asarray(tokens_X)
    left_X = np.asarray(left_X)
    right_X = np.asarray(right_X)
    target_one_hots = np.asarray(target_one_hots)

    return left_X, tokens_X, right_X, target_one_hots, char_vector_dict, train_token_index


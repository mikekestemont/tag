from __future__ import print_function

import ConfigParser
from operator import itemgetter
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.utils import np_utils
from embeddings import EmbeddingsPretrainer
import utils
from model import build_model

font = {'size' : 10}
plt.rc('font', **font)


class Tagger:
    def __init__(self, config_path):
        config = ConfigParser.ConfigParser()
        config.read(config_path)

        # parse the param
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

        # set params:
        self.train_dir = param_dict['train_dir']
        self.dev_dir = param_dict['dev_dir']
        self.test_dir = param_dict['test_dir']
        self.context_representation = param_dict['context_representation']
        self.pretrain_embeddings = param_dict['pretrain_embeddings']
        self.target_representation = param_dict['target_representation']
        self.include_target_one_hot = param_dict['include_target_one_hot']
        self.max_pooling = param_dict['max_pooling']
        self.std_token_len = param_dict['std_token_len']
        self.nb_left_tokens = param_dict['nb_left_tokens']
        self.left_char_len = param_dict['left_char_len']
        self.nb_right_tokens = param_dict['nb_right_tokens']
        self.right_char_len = param_dict['right_char_len']
        self.nb_epochs = param_dict['nb_epochs']
        self.batch_size = param_dict['batch_size']
        self.nb_filters = param_dict['nb_filters']
        self.filter_length = param_dict['filter_length']
        self.nb_dense_dims = param_dict['nb_dense_dims']
        self.embedding_dims = param_dict['embedding_dims']
        print(param_dict)

    def load_data(self, nb_instances, label_idxs):
        self.train_tokens, self.train_labels = \
                utils.load_data_dir(data_dir = self.train_dir,
                                    nb_instances = nb_instances,
                                    label_idxs=label_idxs)
        self.train_token_set = set(self.train_tokens) # for evaluation purposes (known vs unknown)
        if self.dev_dir:
            self.dev_tokens, self.dev_labels = \
                    utils.load_data_dir(data_dir = self.dev_dir,
                                        nb_instances = nb_instances,
                                        label_idxs=label_idxs)
        self.test_tokens, self.test_labels = \
                utils.load_data_dir(data_dir = self.test_dir,
                                    nb_instances = nb_instances,
                                    label_idxs=label_idxs)

    def encode_labels(self):
        # ignore boundary markers
        self.train_labels = [l for l in self.train_labels if l not in ("@", "$")]
        if self.dev_dir:
            self.dev_labels = [l for l in self.dev_labels if l not in ("@", "$")]
        self.test_labels = [l for l in self.test_labels if l not in ("@", "$")]
        
        # fit a labelencoder on all labels:
        self.label_encoder = LabelEncoder()
        if self.dev_dir:
            self.label_encoder.fit(self.train_labels + self.dev_labels + self.test_labels)
        else:
            self.label_encoder.fit(self.train_labels + self.test_labels)
        
        # transform labels to int representation:
        self.train_int_labels = self.label_encoder.transform(self.train_labels)
        if self.dev_dir:
            self.dev_int_labels = self.label_encoder.transform(self.dev_labels)
        self.test_int_labels = self.label_encoder.transform(self.test_labels)
        
        print("> nb distinct train labels:", max(self.train_int_labels)+1)
        
        # convert labels to one-hot represenation for cross-entropy:
        self.train_y = np_utils.to_categorical(self.train_int_labels,
                                               len(self.label_encoder.classes_))
        if self.dev_dir:
            self.dev_y = np_utils.to_categorical(self.dev_int_labels,
                                               len(self.label_encoder.classes_))
        self.test_y = np_utils.to_categorical(self.test_int_labels,
                                               len(self.label_encoder.classes_))

    def vectorize_instances(self, tokens):
        left_X, tokens_X, right_X, target_one_hots = [], [], [], []
        filler = np.zeros(len(self.train_token_index))

        for token_idx, token in enumerate(tokens):
            if token in ("@", "$"):
                continue  # ignore boundary markers:

            if self.include_target_one_hot:
                try:
                    target_one_hots.append([self.train_token_index[token]])
                except KeyError:
                    target_one_hots.append([0])

            # vectorize target token:
            if self.target_representation in \
                    ('flat_characters', 'lstm_characters',\
                     'flat_convolution', 'lstm_convolution'):
                tokens_X.append(utils.vectorize_charseq(seq=token,
                                                        char_vector_dict=self.train_char_vector_dict,
                                                        std_seq_len=self.std_token_len))
            elif target_representation == 'embedding':
                try:
                    tokens_X.append([self.train_token_index[token]])
                except KeyError:
                    tokens_X.append([0])

            # vectorize context:
            if self.context_representation in \
                    ('flat_characters', 'lstm_characters',\
                     'flat_convolution', 'lstm_convolution'):

                # vectorize left context:
                left_context = [tokens[token_idx-(t+1)] for t in range(self.nb_left_tokens)
                                     if token_idx-(t+1) >= 0][::-1]
                left_str = " ".join(left_context)
                left_X.append(utiels.vectorize_charseq(seq=left_str,
                                                       char_vector_dict=self.train_char_vector_dict,
                                                       std_seq_len=self.left_char_len))

                # vectorize right context:
                right_str = " ".join([tokens[token_idx+t+1] for t in range(self.nb_right_tokens)
                                     if token_idx+t+1 < len(tokens)])
                right_X.append(utils.vectorize_charseq(seq=right_str,
                                                       char_vector_dict=self.train_char_vector_dict,
                                                       std_seq_len=self.right_char_len))

            elif self.context_representation in ('flat_embeddings', 'lstm_embeddings'):
                # vectorize left context:
                left_context_tokens = [tokens[token_idx-(t+1)]\
                                            for t in range(self.nb_left_tokens)\
                                                if token_idx-(t+1) >= 0][::-1]
                idxs = []
                if left_context_tokens:
                    idxs = [self.train_token_index[t]\
                                if t in self.train_token_index\
                                else 0 for t in left_context_tokens]
                while len(idxs) < self.nb_left_tokens:
                    idxs = [0] + idxs
                left_X.append(idxs)

                # vectorize right context:
                right_context_tokens = [tokens[token_idx+t+1]\
                                            for t in range(self.nb_right_tokens)\
                                                if token_idx+t+1 < len(tokens)]
                idxs = []
                if right_context_tokens:
                    idxs = [self.train_token_index[t]\
                                if t in self.train_token_index\
                                else 0 for t in right_context_tokens]
                while len(idxs) < self.nb_right_tokens:
                    idxs.append(0)
                right_X.append(idxs)

        tokens_X = np.asarray(tokens_X)
        left_X = np.asarray(left_X)
        right_X = np.asarray(right_X)
        target_one_hots = np.asarray(target_one_hots)

        return left_X, tokens_X, right_X, target_one_hots


    def vectorize_datasets(self):
        # fit dicts etc. on train data
        self.train_char_vector_dict = utils.get_char_vector_dict(self.train_tokens)
        self.train_token_index = utils.get_train_token_index(self.train_tokens)

        # transform training, dev and test data:
        self.train_left_X, self.train_tokens_X,\
        self.train_right_X, self.train_target_one_hots = \
                        self.vectorize_instances(tokens=self.train_tokens)
        if self.dev_dir:
            self.dev_left_X, self.dev_tokens_X,\
            self.dev_right_X, self.dev_target_one_hots = \
                            self.vectorize_instances(tokens=self.dev_tokens)
        self.test_left_X, self.test_tokens_X,\
        self.test_right_X, self.test_target_one_hots = \
                        self.vectorize_instances(tokens=self.test_tokens)

    def set_model(self):
        # get embeddings if necessary:
        self.embeddings = None
        if self.pretrain_embeddings:
            pretrainer = EmbeddingsPretrainer(tokens=self.train_tokens,
                                              size=self.embedding_dims)
            vocab = [k for k,v in sorted(self.train_token_index.items(),\
                                        key=itemgetter(1))]
            self.embeddings = pretrainer.get_weights(vocab)

        self.model = build_model(std_token_len=self.std_token_len,
                                 left_char_len=self.left_char_len,
                                 right_char_len=self.right_char_len,
                                 filter_length=self.filter_length,
                                 nb_filters=self.nb_filters,
                                 nb_dense_dims=self.nb_dense_dims,
                                 nb_left_tokens=self.nb_left_tokens,
                                 nb_right_tokens=self.nb_right_tokens,
                                 nb_labels=len(self.label_encoder.classes_),
                                 char_vector_dict=self.train_char_vector_dict,
                                 train_token_index=self.train_token_index,
                                 target_representation=self.target_representation,
                                 context_representation=self.context_representation,
                                 embedding_dims=self.embedding_dims,
                                 include_target_one_hot=self.include_target_one_hot,
                                 max_pooling=self.max_pooling,
                                 pretrain_embeddings=self.pretrain_embeddings,
                                 embeddings=self.embeddings)
        #for l in self.model.get_weights():
        #    print(l.shape)
    
    def get_baseline(self):
        # Levenshtein baseline:
        if self.dev_dir:
            print("+++ BASELINE SCORE (dev)")
            all_acc, known_acc, unknown_acc, proportion_unknown = utils.baseline(train_tokens = self.train_tokens,
                                                                                 train_labels = self.train_labels,
                                                                                 test_tokens = self.dev_tokens,
                                                                                 test_labels = self.dev_labels)
            print("\t - all acc:\t{:.2%}".format(all_acc))
            print("\t - known acc:\t{:.2%}".format(known_acc))
            print("\t - unknown acc:\t{:.2%}".format(unknown_acc))
            print("\t - proportion unknown:\t{:.2%}".format(proportion_unknown))

        print("+++ BASELINE SCORE (test)")
        all_acc, known_acc, unknown_acc, proportion_unknown = utils.baseline(train_tokens = self.train_tokens,
                                                                             train_labels = self.train_labels,
                                                                             test_tokens = self.test_tokens,
                                                                             test_labels = self.test_labels)
        print("\t - all acc:\t{:.2%}".format(all_acc))
        print("\t - known acc:\t{:.2%}".format(known_acc))
        print("\t - unknown acc:\t{:.2%}".format(unknown_acc))
        print("\t - proportion unknown:\t{:.2%}".format(proportion_unknown))

    def plot_filters(self, filename='filter_weights.tiff'):
        for weights in self.model.get_weights():
            if len(weights.shape) == 4: # conv layer
                weight_tensor = weights
                # get shape info:
                nb_filters = weight_tensor.shape[0]
                nb_chars = weight_tensor.shape[1]
                filter_size = weight_tensor.shape[2]

                chars = list('X'*len(self.train_char_vector_dict))
                for char, vector in self.train_char_vector_dict.items():
                    chars[np.argmax(vector)] = char
                
                for filter_idx, filter_ in enumerate(weight_tensor[:50]):
                    print('\t- filter', filter_idx+1)
                    filter_ = np.squeeze(filter_).transpose()
                    for position, position_weights in enumerate(filter_):
                        info = []
                        for score, char in sorted(zip(position_weights, chars), reverse=True)[:4]:
                            sc = "{:.3}".format(score)
                            info.append(char+' : '+sc)
                        print('\t\t+ pos', position+1, '>', '  |  '.join(info))

    def load_weights(self, filename='weights.hdf5'):
        self.model.load_weights(filename)

    def fit(self, filename='weights.hdf5'):
        train_losses, dev_losses = [], []
        train_accs, dev_accs = [], []
        dev_known_accs, dev_unknown_accs = [], []

        for e in range(self.nb_epochs):
            # save
            self.model.save_weights(filename, overwrite=True)
            # visualize:
            self.plot_filters()
            if self.context_representation != 'None':
                print("-> epoch ", e+1, "...")
                d = {'left_input': self.train_left_X,
                            'target_input': self.train_tokens_X,
                            'right_input': self.train_right_X,
                            'target_one_hot_input': self.train_target_one_hots,
                            'label_output': self.train_y,
                    }
                self.model.fit(data=d,
                               nb_epoch = 1,
                               batch_size = self.batch_size)

                print("+++ TRAIN SCORE")
                train_loss = self.model.evaluate(data=d, batch_size = self.batch_size)
                print("\t - loss:\t{:.3}".format(train_loss))
                train_losses.append(train_loss)

                train_predictions = self.model.predict({'left_input': self.train_left_X,
                                          'target_input': self.train_tokens_X,
                                          'target_one_hot_input': self.train_target_one_hots,
                                          'right_input': self.train_right_X,
                                         },
                                         batch_size = self.batch_size)
                train_all_acc, _, _ = utils.accuracies(predictions = train_predictions,
                                                                   gold_labels = self.train_int_labels,
                                                                   test_tokens = self.train_tokens,
                                                                   train_token_set = self.train_token_set)
                print("\t - all acc:\t{:.2%}".format(train_all_acc))
                train_accs.append(train_all_acc)
                
                if self.dev_dir:
                    print("+++ DEV SCORE")
                    d = {'left_input': self.dev_left_X,
                         'target_input': self.dev_tokens_X,
                         'right_input': self.dev_right_X,
                         'target_one_hot_input': self.dev_target_one_hots,
                        }
                    dev_predictions = self.model.predict(data=d, batch_size = self.batch_size)
                    d['label_output'] = self.dev_y
                    dev_loss = self.model.evaluate(data=d, batch_size = self.batch_size)
                    print("\t - loss:\t{:.3}".format(dev_loss))
                    dev_losses.append(dev_loss)
                    dev_all_acc, dev_known_acc, dev_unknown_acc = utils.accuracies(predictions = dev_predictions,
                                                                       gold_labels = self.dev_int_labels,
                                                                       test_tokens = self.dev_tokens,
                                                                       train_token_set = self.train_token_set)
                    dev_accs.append(dev_all_acc)
                    dev_known_accs.append(dev_known_acc)
                    dev_unknown_accs.append(dev_unknown_acc)

                    print("\t - all acc:\t{:.2%}".format(dev_all_acc))
                    print("\t - known acc:\t{:.2%}".format(dev_known_acc))
                    print("\t - unknown acc:\t{:.2%}".format(dev_unknown_acc))

                print("+++ TEST SCORE")
                test_predictions = self.model.predict({'left_input': self.test_left_X,
                                          'target_input': self.test_tokens_X,
                                          'right_input': self.test_right_X,
                                          'target_one_hot_input': self.test_target_one_hots,
                                         },
                                         batch_size = self.batch_size)
                test_all_acc, test_known_acc, test_unknown_acc = utils.accuracies(predictions = test_predictions,
                                                                   gold_labels = self.test_int_labels,
                                                                   test_tokens = self.test_tokens,
                                                                   train_token_set = self.train_token_set)
                print("\t - all acc:\t{:.2%}".format(test_all_acc))
                print("\t - known acc:\t{:.2%}".format(test_known_acc))
                print("\t - unknown acc:\t{:.2%}".format(test_unknown_acc))

                plt.clf()
                f, ax = plt.subplots(1,1)
                plt.tick_params(axis='both', which='major', labelsize=6)
                plt.tick_params(axis='both', which='minor', labelsize=6)
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                ax.plot(train_losses, c='b')
                if self.dev_dir:
                    ax.plot(dev_losses, c='g')
                ax2 = plt.gca().twinx()
                plt.tick_params(axis='both', which='major', labelsize=6)
                plt.tick_params(axis='both', which='minor', labelsize=6)
                ax2.plot(train_accs, label='Train Acc (all)', c='r')
                if self.dev_dir:
                    ax2.plot(dev_accs, label='Devel Acc (all)', c='c')
                    ax2.plot(dev_known_accs, label='Devel Acc (kno)', c='m')
                    ax2.plot(dev_unknown_accs, label='Devel Acc (unk)',  c='y')
                # hack: add dummy legend items;
                ax2.plot(0, 0, label='Train Loss', c='b')
                if self.dev_dir:
                    ax2.plot(0, 0, label='Devel Loss', c='g')
                plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                            ncol=3, mode="expand", borderaxespad=0., fontsize = 'x-small')
                ax2.set_ylabel('Accuracy')
                plt.savefig('losses.pdf')

                with open('losses.tsv', 'w') as f:
                    if self.dev_dir:
                        f.write('idx\ttrain\tdev\n')
                        for i in range(len(train_losses)):
                            f.write('\t'.join((str(i+1), str(train_losses[i]), str(dev_losses[i])))+'\n')
                    else:
                        f.write('idx\ttrain\n')
                        for i in range(len(train_losses)):
                            f.write('\t'.join((str(i+1), str(train_losses[i])))+'\n')

                with open('accs.tsv', 'w') as f:
                    if self.dev_dir:
                        f.write('idx\ttrain_all\tdev_all\tdev_known\tdev_unknown\n')
                        for i in range(len(train_losses)):
                            f.write('\t'.join((str(i+1), str(train_accs[i]),\
                                                str(dev_accs[i]), str(dev_known_accs[i]),\
                                                str(dev_unknown_accs[i])))+'\n')
                    else:
                        f.write('idx\ttrain_all\n')
                        for i in range(len(train_losses)):
                            f.write('\t'.join((str(i+1), str(train_accs[i])))+'\n')

            elif self.context_representation == "None":
                print("-> epoch ", e+1, "...")
                self.model.fit({'target_input': self.train_tokens_X,
                       'target_one_hot_input': self.train_target_one_hots,
                        'label_output': self.train_y
                      },
                            nb_epoch = 1,
                            batch_size = self.batch_size)
                
                print("+++ TRAIN SCORE")
                train_predictions = self.model.predict({'target_input': self.train_tokens_X,
                                         'target_one_hot_input': self.train_target_one_hots},
                                         batch_size = self.batch_size)
                all_acc, _, _ = utils.accuracies(predictions = train_predictions,
                                                                   gold_labels = self.train_int_labels,
                                                                   test_tokens = self.train_tokens,
                                                                   train_token_set = self.train_token_set)
                print("\t - all acc:\t{:.2%}".format(all_acc))
                
                if self.dev_dir:
                    print("+++ DEV SCORE")
                    dev_predictions = self.model.predict({'target_input': self.dev_tokens_X,
                                             'target_one_hot_input': self.dev_target_one_hots,},
                                             batch_size = self.batch_size)
                    all_acc, known_acc, unknown_acc = utils.accuracies(predictions = dev_predictions,
                                                                       gold_labels = self.dev_int_labels,
                                                                       test_tokens = self.dev_tokens,
                                                                       train_token_set = self.train_token_set)
                    print("\t - all acc:\t{:.2%}".format(all_acc))
                    print("\t - known acc:\t{:.2%}".format(known_acc))
                    print("\t - unknown acc:\t{:.2%}".format(unknown_acc))

                print("+++ TEST SCORE")
                test_predictions = self.model.predict({'target_input': self.test_tokens_X,
                                         'target_one_hot_input': self.test_target_one_hots},
                                         batch_size = self.batch_size)
                all_acc, known_acc, unknown_acc = utils.accuracies(predictions = test_predictions,
                                                                   gold_labels = self.test_int_labels,
                                                                   test_tokens = self.test_tokens,
                                                                   train_token_set = self.train_token_set)
                print("\t - all acc:\t{:.2%}".format(all_acc))
                print("\t - known acc:\t{:.2%}".format(known_acc))
                print("\t - unknown acc:\t{:.2%}".format(unknown_acc))

                #self.plot_filters()
        
        self.model.save_weights('weights.hdf5')









from __future__ import print_function

import os
import sys
import logging
from operator import itemgetter

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

import utils
import model
from tagger import Tagger

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

def main():
    print('::: started :::')

    model_path = os.path.abspath(sys.argv[1])
    param_dict = dict()
    
    config_path = model_path+'/config.txt'
    param_dict = utils.parse_cmd_line(config_path)
    print("> using config file: "+str(config_path))
    
    tagger = Tagger(config_path=config_path)
    tagger.load_data(nb_instances=100000000000000, label_idxs=[2])
    tagger.encode_labels(load_pickles=True)
    tagger.vectorize_datasets(load_pickles=True)
    tagger.set_model(load_pickles=True)
    tagger.load_weights()
    tagger.test()
    print('::: ended :::')


if __name__ == '__main__':
    main()
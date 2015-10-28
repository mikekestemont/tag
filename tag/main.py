from __future__ import print_function

import os
import utils
import sys
import logging
from operator import itemgetter

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

import model

from tagger import Tagger

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

def main():
    print('::: started :::')

    param_dict = dict()
    config_path = os.path.abspath(sys.argv[1])
    print("> using config file: "+str(config_path))
    if config_path:
        param_dict = utils.parse_cmd_line(config_path)
    else:
        raise ValueError("No config file specified.")
    
    tagger = Tagger(config_path)
    tagger.load_data(nb_instances=10000000000, label_idxs=[2])
    tagger.encode_labels()
    tagger.vectorize_datasets()
    tagger.set_model()
    #tagger.load_weights()
    tagger.fit()
    
    print('::: ended :::')

if __name__ == '__main__':
    main()
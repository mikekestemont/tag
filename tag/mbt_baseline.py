from __future__ import print_function

import os
import sys
import logging
import subprocess

import utils

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

def dir_to_mbt_format(input_dir, outfilename, nb_instances):
    tokens, labels = utils.load_data_dir(data_dir=input_dir,
                                                    nb_instances=nb_instances,
                                                    label_idxs=[2])
    with open(outfilename, 'w') as f:
        for token, label in zip(tokens, labels):
            if token in ('$', '@') or label in ('$', '@'):
                f.write('<utt>\n')
            else:
                f.write('\t'.join((token, label))+'\n')

def main():
    print('::: started :::')

    param_dict = dict()
    config_path = os.path.abspath(sys.argv[1])
    if config_path:
        param_dict = utils.parse_cmd_line(config_path)
    else:
        raise ValueError("No config file specified.")
    print("> using config file: "+str(config_path))

    nb_instances = 1000000000000000

    mbt_path = '../mbt_workspace/'
    if not os.path.isdir(mbt_path):
        os.mkdir(mbt_path)

    # reformat train data:
    dir_to_mbt_format(input_dir=param_dict['train_dir'],
                    outfilename=mbt_path+'/train.txt',
                    nb_instances=nb_instances)

    # reformat dev data:
    dir_to_mbt_format(input_dir=param_dict['dev_dir'],
                    outfilename=mbt_path+'/dev.txt',
                    nb_instances=nb_instances)

    # reformat test data:
    dir_to_mbt_format(input_dir=param_dict['test_dir'],
                    outfilename=mbt_path+'/test.txt',
                    nb_instances=nb_instances)

    # train the model:
    subprocess.call('Mbtg -T '+mbt_path+'train.txt', shell=True)

    # first, evaluate on dev:
    subprocess.call('Mbt -s '+mbt_path+'train.txt.settings -T '+mbt_path+'dev.txt', shell=True)

    # then, evaluate on test:
    subprocess.call('Mbt -s '+mbt_path+'train.txt.settings -T '+mbt_path+'test.txt', shell=True)

    print('::: ended :::')

if __name__ == '__main__':
    main()

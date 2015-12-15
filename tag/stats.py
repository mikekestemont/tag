from __future__ import print_function
import glob
import codecs
import pandas as pd
import numpy as np

def parse(p):
    data = []
    for line in open(p, 'r'):
        line = line.strip()
        if not line or line.startswith('@'):
            continue
        else:
            try:
                token, pos, lemma = line.split('\t')
                token = token.replace('~', '')
                token = token.replace(' ', '')
                data.append((token, pos, lemma))
            except ValueError:
                print(line)
    return data

def basic_stats(data):
    tokens = [d[0] for d in data]
    lemmas = [d[2] for d in data]
    row = []
    # how many items?
    row.append(len(data))
    # distinct tokens?
    row.append(len(set(tokens)))
    # distinct lemmas?
    row.append(len(set(lemmas)))
    # average nb of tokens per lemmas?
    cnts = {}
    for tok, _, lem in data:
        try:
            cnts[lem].add(tok)
        except KeyError:
            cnts[lem] = set()
            cnts[lem].add(tok)
    m = round(np.mean([len(cnts[d]) for d in cnts]), 2)
    row.append(m)
    return row

def unknown_stats(dev_data, train_token_set, train_lemma_set):
    unknown = 0.0
    upperbound = 0.0
    for token, _, lemma in dev_data:
        if token not in train_token_set:
            unknown += 1
            if lemma in train_lemma_set:
                upperbound += 1
    return round(unknown / len(dev_data)*100, 2), round(upperbound / unknown*100, 2)


def main():
    df = pd.DataFrame(columns=['corpus', 'split', 'nb words', 'unique tokens', 'unique lemmas', 'nb tokens per lemma', 'perc. unknown tokens', 'upperbound unknown'])
    print('Collecting stats...')

    for corpus in ('relig', 'crm-adelheid', 'cg-lit', 'cg-admin'):
        corpus_dir = '../data/'+corpus

        # get train stats:
        f = glob.glob(corpus_dir+'/train/*.3col')[0]
        train_data = parse(f)

        train_tokens = set(d[0] for d in train_data)
        train_lemmas = set(d[2] for d in train_data)

        # which subpart?
        row = [corpus, 'train']
        row.extend(basic_stats(train_data))    
        # no unknown tokens in train:
        row.append('NA')
        # no upperbound in train:
        row.append('NA')
        df.loc[len(df)] = row
        

        # get dev stats:
        f = glob.glob(corpus_dir+'/dev/*.3col')[0]
        dev_data = parse(f)

        # which subpart?
        row = [corpus, 'dev']
        row.extend(basic_stats(dev_data))
        dev_tokens = [d[0] for d in dev_data]
        
        row.extend(unknown_stats(dev_data, train_token_set=train_tokens, train_lemma_set=train_lemmas))
        df.loc[len(df)] = row

        # get test stats:
        f = glob.glob(corpus_dir+'/test/*.3col')[0]
        test_data = parse(f)

        # which subpart?
        row = [corpus, 'test']
        row.extend(basic_stats(test_data))
        test_tokens = [d[0] for d in test_data]
        
        row.extend(unknown_stats(test_data, train_token_set=train_tokens, train_lemma_set=train_lemmas))
        df.loc[len(df)] = row

    df = df.set_index('corpus')
    df.to_csv('stats.csv')
        



if __name__ == '__main__':
    main()
import os
import shutil
import codecs

def clean_afrodata():
    for language in os.listdir('/home/mike/afrodata'):
        print(language)
        try:
            shutil.rmtree('/home/mike/GitRepos/Midas/data/uniform/annotated/'+language)
        except:
            pass
        os.mkdir('/home/mike/GitRepos/Midas/data/uniform/annotated/'+language)
        os.mkdir('/home/mike/GitRepos/Midas/data/uniform/annotated/'+language+'/train')
        os.mkdir('/home/mike/GitRepos/Midas/data/uniform/annotated/'+language+'/dev')
        os.mkdir('/home/mike/GitRepos/Midas/data/uniform/annotated/'+language+'/test')
        train_file = codecs.open('/home/mike/GitRepos/Midas/data/uniform/annotated/'+language+'/train/'+language+'_train.3col', 'w+', encoding='utf8')
        dev_file = codecs.open('/home/mike/GitRepos/Midas/data/uniform/annotated/'+language+'/dev/'+language+'_dev.3col', 'w+', encoding='utf8')
        test_file = codecs.open('/home/mike/GitRepos/Midas/data/uniform/annotated/'+language+'/test/'+language+'_test.3col', 'w+', encoding='utf8')

        for fold in os.listdir('/home/mike/afrodata/'+language):
            if fold.endswith('9'):
                F = test_file
            elif fold.endswith('8'):
                F = dev_file
            else:
                F = train_file
            for line in codecs.open('/home/mike/afrodata/'+language+'/'+fold, 'r', encoding='utf8'):
                for item in line.strip().split():
                    F.write('\t'.join(item.split('_'))+'\n')
                F.write('\n')

        train_file.close()
        dev_file.close()
        test_file.close()

def split_french_data(orig_file='/Users/mike/Desktop/nca3_tabular.3col'):
    if not os.path.isdir('../data/old_french'):
        os.mkdir('../data/old_french')
    if not os.path.isdir('../data/old_french/train'):
        os.mkdir('../data/old_french/train')
    train_file = open('../data/old_french/train/old_french_train.3col', 'w+')
    if not os.path.isdir('../data/old_french/dev'):
        os.mkdir('../data/old_french/dev')
    dev_file = open('../data/old_french/dev/old_french_dev.3col', 'w+')
    if not os.path.isdir('../data/old_french/test'):
        os.mkdir('../data/old_french/test')
    test_file = open('../data/old_french/test/old_french_test.3col', 'w+')
    items = []
    for line in codecs.open(orig_file):
        line = line.strip()
        if line.startswith('@'):
            title = line[1:]
            print(title)
            # flush old ones
            fold_size = int(len(items)/float(10))
            if items:
                train_items = items[0:fold_size*8]
                dev_items = items[fold_size*8:fold_size*9]
                test_items = items[fold_size*9:fold_size*10]
                train_file.write('@'+title+'\n')
                dev_file.write('@'+title+'\n')
                test_file.write('@'+title+'\n')
                for t in train_items:
                    train_file.write('\t'.join(t)+'\n')
                for t in dev_items:
                    dev_file.write('\t'.join(t)+'\n')
                for t in test_items:
                    test_file.write('\t'.join(t)+'\n')
            items = []
        else:
            tok, pos, _ = line.split('\t')
            items.append((tok, pos))
    train_file.close()
    dev_file.close()
    test_file.close()


split_french_data()
        
import os
import time
import math
import numpy as np

import pdb

def INFO_LOG(info):
    print "[%s]%s" % (time.strftime("%Y-%m-%d %X", time.localtime()), info)

class Vocab(object):
    def __init__(self):
        self.BOS = "<s>"
        self.EOS = "</s>"
        self.UNK = "<unk>"

    def buildFromFiles(self, files):
        if type(files) is not list:
            raise ValueError("buildFromFiles input type error")

        INFO_LOG("build vocabulary from files ...")
        self.word_cnt = {self.EOS: 0}
        for _file in files:
            line_num = 0
            for line in open(_file):
                line_num += 1
                for w in line.strip().replace('<UNK>', self.UNK).split():
                    if self.word_cnt.has_key(w):
                        self.word_cnt[w] += 1
                    else:
                        self.word_cnt[w] = 1
            #self.word_cnt[self.BOS] += line_num
            self.word_cnt[self.EOS] += line_num
        count_pairs = sorted(self.word_cnt.items(), key = lambda x: (-x[1], x[0]))
        self.words, _ = list(zip(*count_pairs))

        #2 ids
        rows = int(math.ceil(math.sqrt(len(self.words))))
        cols = int(math.ceil(math.sqrt(len(self.words))))
        ids = []

        for row in xrange(rows):
            for col in xrange(cols):
                ids.append([row, col])

        final_ids = ids[0:len(self.words)]
        self.word2id = dict(zip(self.words, final_ids))
        self.UNK_ID = self.word2id[self.UNK]
        INFO_LOG("vocab size: {}".format(self.size()))
 
    def encode(self, sentence):
        return [self.word2id[w] if self.word2id.has_key(w) else self.UNK_ID for w in sentence]

    def decode(self, ids):
        return [self.words[_id] for _id in ids]

    def size(self):
        return len(self.words)

    def word2id(self):
        return self.word2id

class Reader(object):
    def __init__(self, data_path):
        self.train_file = os.path.join(data_path, 'ptb.train.txt')
        self.valid_file = os.path.join(data_path, 'ptb.valid.txt')
        self.test_file = os.path.join(data_path, 'ptb.test.txt')

        self.vocab = Vocab()
        self.vocab.buildFromFiles([self.train_file])

    def getVocabSize(self):
        return self.vocab.size()

    def get_word_dict(self):
        return self.vocab.word2id

    def yieldSpliceBatch(self, tag, batch_size, step_size, worddict):
        eos_index = worddict[self.vocab.EOS]
        unk_index = worddict[self.vocab.UNK]
        if tag == 'Train':
            _file = self.train_file
        elif tag == 'Valid':
            _file = self.valid_file
        else:
            _file = self.test_file

        INFO_LOG("File: %s" % _file)
        data = []
        line_num = 0
        for line in open(_file):
            tokens = line.strip().split()
            #pdb.set_trace()
            data += [worddict[w] if worddict.has_key(w) else unk_index for w in tokens] + [eos_index]
            line_num += 1
        total_token = len(data)
        token_num = (total_token - line_num)

        data_len = len(data)
        batch_len = data_len // batch_size
        batch_num = (batch_len - 1) // step_size
        if batch_num == 0:
            raise ValueError("batch_num == 0, decrease batch_size or step_size")
        
        INFO_LOG("  {} sentence, {}/{} tokens with/out {}".format(line_num, total_token, token_num, self.vocab.EOS))

        used_token = batch_num * batch_size * step_size
        INFO_LOG("  {} batches, {}*{}*{} = {}({:.2%}) tokens will be used".format(batch_num,  
            batch_num, batch_size, step_size, used_token, float(used_token) / total_token))

        word_data = np.zeros([batch_size, batch_len, 2], dtype=np.int32)
        for j in range(batch_size):
            index = j * batch_len
            #pdb.set_trace()
            word_data[j] = data[index : index + batch_len]
        for batch_id in range(batch_num):
            index = step_size * batch_id
            x = word_data[:, index : index + step_size]
            y = word_data[:, index + 1 : index + step_size + 1]
            n = batch_size * step_size
            yield(batch_id, batch_num, x, y, n) 


import os
import pickle
import random
import logging

import numpy as np


def load_word2vec(path, id_to_vec):
    with open(path, "rb") as f:
        word_vec = pickle.load(f)
        word2vec = []
        for i, word in id_to_vec.items():
            if word in word_vec:
                word2vec.append(word_vec[word])
            else:
                word2vec.append(word_vec["<UNK>"])
    return word2vec


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join("./log", name + ".log"))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def test_ner(results, path):
    script_file = "./conlleval"
    output_file = os.path.join(path, "ner_predict.utf8")
    result_file = os.path.join(path, "ner_result.utf8")
    with open(output_file, "w") as f:
        to_write = []
        for block in results:
            for line in block[0]:
                to_write.append(line + "\n")
            if block[1]:
                to_write.append("\n")

        f.writelines(to_write)
    os.system("perl {} < {} > {}".format(script_file, output_file, result_file))
    eval_lines = []
    with open(result_file) as f:
        for line in f:
            eval_lines.append(line.strip())
    return eval_lines


def calculate_accuracy(labels, paths, lengths):
    # calculate token level accuracy, return correct tag numbers and total tag numbers
    total = 0
    correct = 0
    for label, path, length in zip(labels, paths, lengths):
        gold = label[length]
        correct += np.sum(np.equal(gold, path))
        total += length
    return correct, total


class BatchManager(object):

    def __init__(self, data, num_tag, word_max_len, batch_size):
        self.data = data
        self.numbatch = len(self.data) // batch_size
        self.batch_size = batch_size
        self.batch_index = 0
        self.len_data = len(data)
        self.num_tag = num_tag

    @staticmethod
    def unpack(data):
        words = []
        tags = []
        lengths = []
        features = []
        str_lines = []
        end_of_doc = []
        for item in data:
            if item["len"] < 0:
                continue
            words.append(item["words"])
            tags.append(item["tags"])
            lengths.append(item["len"])
            features.append(item["features"])
            str_lines.append(item["str_line"])
            end_of_doc.append(item["end_of_doc"])
        return {"words": words,
                "tags": tags,
                "len": lengths,
                "features": features,
                "str_lines": str_lines,
                "end_of_doc": end_of_doc}

    def shuffle(self):
        random.shuffle(self.data)

    def iter_batch(self):
        for i in range(self.numbatch+1):
            if i == self.numbatch:
                data = self.data[i*self.batch_size:]
            else:
                data = self.data[i*self.batch_size:(i+1)*self.batch_size]
            yield self.unpack(data)


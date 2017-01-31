# import codecs
#
#
# def read_conll_file(path):
#     """
#     This function will Load sentences in Conll format.
#     A line must contain at least a word and its tag.
#     Sentences are separated by empty lines.
#     """
#     sentences = []
#     sentence = []
#     for line in codecs.open(path, "r", "utf8"):
#         line = line.rstrip()
#         if not line:
#             if len(sentence) > 0:
#                 if "DOCSTART" not in sentence[0][0]:
#                     sentences.append(sentence)
#                 sentence = []
#         else:
#             word = line.split(" ")
#             if word:
#                 assert len(word) > 1, word
#                 sentence.append(word)
#     if len(sentence) > 0:
#         if "DOCSTART" not in sentence[0][0]:
#             sentences.append(sentence)
#     return sentences
#
#
# def doc_to_sentence(doc, max_len):
#     """
#     This function will cut doc to sentences with ！|。|；,
#     If sentence is longer than max_len, it will be cut with ，|、|／
#     The function return a list of integers with length of each sentence
#     """
#     pattern1 = "。！|；"
#     pattern2 = "，、／"
#     pre_index = -1
#     pattern_index = -1
#     sentences = []
#     sentence = []
#     for i, line in enumerate(doc):
#         sentence.append(line)
#         if i - pre_index > max_len-1:
#             if pattern_index > pre_index:
#                 sentences.append(sentence[:pattern_index-pre_index])
#                 sentence = sentence[pattern_index-pre_index:]
#                 pre_index = pattern_index
#             else:
#                 pre_index = i-1
#                 sentences.append(sentence)
#                 sentence = []
#         else:
#             if line[0] in pattern2:
#                 pattern_index = i
#                 if line[0] in pattern1:
#                     sentences.append(sentence)
#                     sentence = []
#                     pre_index = i
#     if sentence:
#         sentences.append(sentence)
#     return sentences
#
#
# def word_mapping(sentences, min_freq):
#     vocab = dict()
#     word_to_id = dict()
#     word_id = 0
#     for sentence in sentences:
#         for word in sentence[0]:
#             if word in word_to_id:
#                 continue
#             if word in vocab:
#                 vocab[word] += 1
#                 if vocab[word] >= min_freq:
#                     word_to_id[word] = word_id
#                     word_id += 1
#             else:
#                 vocab[word] = 1
#     word_to_id["<UNK>"] = word_id
#     word_to_id["<PAD>"] = word_id+1
#
#     return word_to_id, {v: k for k, v in word_to_id.items()}
#
#
# def prepare_data(data, word_to_id, tag_to_id, max_words):
#     processed_data = []
#     for doc in data:
#         doc = doc_to_sentence(doc, max_words)
#         len_doc = len(doc)
#         for i, sentence in enumerate(doc):
#             len_sen = len(sentence)
#             str_words = []
#             words = []
#             tags = []
#             for line in sentence:
#                 word = line[0].lower()
#                 str_words.append(word)
#                 words.append(word_to_id[word] if word in word_to_id
#                              else word_to_id["<UNK>"])
#                 tags.append(tag_to_id[line[-1]])
#             words += [word_to_id["<PAD>"]] * (max_words-len_sen)
#             tags += [tag_to_id["O"]] * (max_words-len_sen)
#             processed_data.append({"str_words": str_words,
#                                    "words": words,
#                                    "tags": tags,
#                                    "len": len_sen,
#                                    "end_of_sentence": i == len_doc-1})
#     return processed_data
#
#
# def load_data(params, tag_to_id):
#     train_file = read_conll_file(params.train_file)
#     dev_file = read_conll_file(params.dev_file)
#     test_file = read_conll_file(params.test_file)
#
#     word_to_id, id_to_word = word_mapping(train_file, params.min_freq)
#     train_data = prepare_data(train_file, word_to_id, tag_to_id, params.word_max_len)
#     dev_data = prepare_data(dev_file, word_to_id, tag_to_id, params.word_max_len)
#     test_data = prepare_data(test_file, word_to_id, tag_to_id, params.word_max_len)
#     return word_to_id, {v: k for k, v in tag_to_id.items()}, train_data, dev_data, test_data
import codecs
import jieba
import numpy as np

def read_conll_file(path):
    """
    This function will Load sentences in Conll format.
    A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    for line in codecs.open(path, "r", "utf8"):
        line = line.rstrip()
        if not line:
            if len(sentence) > 0:
                if "DOCSTART" not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split(" ")
            if word:
                assert len(word) > 1, word
                sentence.append(word)
    if len(sentence) > 0:
        if "DOCSTART" not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


def doc_to_sentence(doc, max_len):
    """
    This function will cut doc to sentences with ！|。|；,
    If sentence is longer than max_len, it will be cut with ，|、|／
    The function return a list of integers with length of each sentence
    """
    pattern1 = "。！|？；"
    pattern2 = "，、／。；"
    pre_index = -1
    pattern_index = -1
    sentences = []
    sentence = []
    for i, line in enumerate(doc):
        sentence.append(line)
        if i - pre_index > max_len-1:
            if pattern_index > pre_index:
                sentences.append(sentence[:pattern_index-pre_index])
                sentence = sentence[pattern_index-pre_index:]
                pre_index = pattern_index
            else:
                pre_index = i-1
                sentences.append(sentence)
                sentence = []
        else:
            if line[0] in pattern2:
                pattern_index = i
                if line[0] in pattern1:
                    sentences.append(sentence)
                    sentence = []
                    pre_index = i
    if sentence:
        sentences.append(sentence)
    return sentences


def word_mapping(data, min_freq):
    vocab = dict()
    word_to_id = dict()
    word_id = 0
    for doc in data:
        for line in doc:
            word = line[0]
            if word not in word_to_id:
                if word in vocab:
                    vocab[word] += 1
                    if vocab[word] >= min_freq:
                        word_to_id[word] = word_id
                        word_id += 1
                else:
                    vocab[word] = 1
    word_to_id["<UNK>"] = word_id
    word_to_id["<PAD>"] = word_id+1
    return word_to_id, {v: k for k, v in word_to_id.items()}


def prepare_data(data, word_to_id, tag_to_id, max_words):
    processed_data = []
    for doc in data:
        doc = doc_to_sentence(doc, max_words)
        len_doc = len(doc)

        for i, sentence in enumerate(doc):
            len_sen = len(sentence)
            str_words = []
            words = []
            tags = []
            for line in sentence:
                word = line[0].lower()
                str_words.append(word)
                words.append(word_to_id[word] if word in word_to_id
                             else word_to_id["<UNK>"])
                tags.append(tag_to_id[line[-1]])
            words += [word_to_id["<PAD>"]] * (max_words-len_sen)
            tags += [tag_to_id["O"]] * (max_words-len_sen)
            features = np.zeros([max_words, 4],dtype=np.float32)
            index = 0
            # BIES tags
            for word in jieba.cut("".join(str_words)):
                len_word = len(word)
                if len_word == 1:
                    features[index, 0] = 1 #S
                    index += 1
                else:
                    features[index, 1] = 1
                    index += 1
                    for i_ in range(len_word-2):
                        features[index, 2] = 1
                        index += 1
                    features[index, 3] = 1
                    index += 1
            processed_data.append({"str_line": str_words,
                                   "words": words,
                                   "tags": tags,
                                   "len": len_sen,
                                   "features": features,
                                   "end_of_doc": i == len_doc-1})
    return processed_data


def load_data(params, tag_to_id):
    train_file = read_conll_file(params.train_file)
    dev_file = read_conll_file(params.dev_file)
    test_file = read_conll_file(params.test_file)
    word_to_id, id_to_word = word_mapping(train_file, params.min_freq)
    train_data = prepare_data(train_file, word_to_id, tag_to_id, params.word_max_len)
    dev_data = prepare_data(dev_file, word_to_id, tag_to_id, params.word_max_len)
    test_data = prepare_data(test_file, word_to_id, tag_to_id, params.word_max_len)
    print(len(train_file))
    print(len(train_data))
    return word_to_id, {v: k for k, v in tag_to_id.items()}, train_data, dev_data, test_data


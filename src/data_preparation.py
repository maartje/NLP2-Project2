import torch
import filepaths as fp
from data_processing import preprocess

SOS_token = 0
EOS_token = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def prepare_training_data(spath, tpath, useCache = True):
    (slang, stensors) = _prepare_training_data_lang(spath, useCache)
    (tlang, ttensors) = _prepare_training_data_lang(tpath, useCache)
    tensor_pairs = list(zip(stensors, ttensors))
    return (slang, tlang, tensor_pairs)

def _prepare_training_data_lang(path, useCache = True):
    path_preprocessed = preprocess(path, useCache)
    lang = _read_language(path_preprocessed)
    tensors = list(_build_tensors(lang, path_preprocessed))
    return (lang, tensors)

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def _read_language(fpath):
    lname = fpath[-2:]
    lang = Lang(lname)

    # TODO load/save?
    with open(fpath, 'r') as lines:
        for line in lines:
            lang.addSentence(line)
    return lang

def _build_tensors(lang, fpath):
    # TODO load/save?
    with open(fpath, 'r') as lines:
        for line in lines:
            yield tensorFromSentence(lang, line)

def _indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = _indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def sentenceFromIndexes(lang, indices):
    words = [lang.index2word[index] for index in indices]
    return ' '.join(words[:-1])


import filepaths as fp
from data_processing import preprocess

SOS_token = 0
EOS_token = 1

def prepare_training_data(spath, tpath, useCache = True):
    (slang, s_index_arrays) = _prepare_training_data_lang(spath, useCache)
    (tlang, t_index_arrays) = _prepare_training_data_lang(tpath, useCache)
    index_array_pairs = list(zip(s_index_arrays, t_index_arrays))
    return (slang, tlang, index_array_pairs)

def _prepare_training_data_lang(path, useCache = True):
    path_preprocessed = preprocess(path, useCache)
    lang = _read_language(path_preprocessed)
    index_arrays = list(_build_index_arrays(lang, path_preprocessed))
    return (lang, index_arrays)

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

def _build_index_arrays(lang, fpath):
    # TODO load/save?
    with open(fpath, 'r') as lines:
        for line in lines:
            yield indexesFromSentence(lang, line)

def indexesFromSentence(lang, sentence):
    indexes = [lang.word2index[word] for word in sentence.split(' ')]
    indexes.append(EOS_token)
    return indexes

def sentenceFromIndexes(lang, indices):
    words = [lang.index2word[index] for index in indices]
    return ' '.join(words[:-1])

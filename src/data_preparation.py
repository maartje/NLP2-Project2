import filepaths as fp

SOS_token = 0
EOS_token = 1
PAD_token =  2

def prepare_data(spath_pp, tpath_pp, spath_test_pp, add_padding = True):
    (slang, s_index_arrays) = _prepare_training_data_lang(spath_pp)
    (tlang, t_index_arrays) = _prepare_training_data_lang(tpath_pp)
    s_index_arrays_test = _prepare_test_data(slang, spath_test_pp)
    
    # calculate the maximum number of segments created by BPE
    max_bpe_length = max([ len(l) for l in s_index_arrays_test + s_index_arrays]) 
    
    # add padding to source sentences
    if add_padding:
        for indices in s_index_arrays:
            add_padding(indices, max_bpe_length)
        for indices_test in s_index_arrays_test:
            add_padding(indices_test, max_bpe_length)
        assert len(s_index_arrays[0]) == len(s_index_arrays[3]) # padding thus same length
        assert len(s_index_arrays[2]) == len(s_index_arrays_test[4]) # padding thus same length

    # zip training lists into pairs
    assert len(s_index_arrays) == len(t_index_arrays)
    index_array_pairs = list(zip(s_index_arrays, t_index_arrays))

    # TODO cache in file?
    return (slang, tlang, index_array_pairs, s_index_arrays_test, max_bpe_length)

def _prepare_test_data(lang, path_preprocessed):
    index_arrays = list(_build_index_arrays(lang, path_preprocessed))
    return index_arrays

def _prepare_training_data_lang(path_preprocessed):
    lang = _read_language(path_preprocessed)
    index_arrays = list(_build_index_arrays(lang, path_preprocessed))
    return (lang, index_arrays)

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2:"PAD"}
        self.n_words = 3  # Count SOS and EOS

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

    with open(fpath, 'r') as lines:
        for line in lines:
            lang.addSentence(line)
    return lang

def _build_index_arrays(lang, fpath):
    with open(fpath, 'r') as lines:
        for line in lines:
            yield indexesFromSentence(lang, line)

def indexesFromSentence(lang, sentence):
    indexes = [lang.word2index[word] for word in sentence.split(' ')]
    indexes.append(EOS_token)
    return indexes

def add_padding(indexes, max_bpe_length):
    indexes += [PAD_token] * (max_bpe_length - len(indexes))
    

def wordsFromIndexes(lang, indices):
    return [lang.index2word[index] for index in indices]

def sentenceFromIndexes(lang, indices):
    words = wordsFromIndexes(lang, indices)
    return ' '.join(words[:-1])

def sentenceFromIndexes_all(lang, indices_all):
    return [sentenceFromIndexes(lang, indices) for indices in indices_all]

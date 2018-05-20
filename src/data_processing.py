import filepaths as fp
import os
from collections import Counter

def preprocess(
        spath_train, tpath_train, spath_test, tpath_test,
        replace_unknown_words = True, useCache = False):

    spath_train_pp = fp.path_to_outputfile(spath_train, '.preprocessed-train')
    tpath_train_pp = fp.path_to_outputfile(tpath_train, '.preprocessed-train')
    spath_test_pp = fp.path_to_outputfile(spath_test, '.preprocessed-test')
    tpath_test_pp = fp.path_to_outputfile(tpath_test, '.preprocessed-test')

    # TODO: use cache?
    # if not(useCache) or (not os.path.isfile(path_to_preprocessed)):
    # else:
    #     print (f'info: uses cached preprocessed file {path_to_preprocessed}')
 
    svocab = preprocess_file(spath_train, spath_train_pp, replace_unknown_words)
    preprocess_file(tpath_train, tpath_train_pp, replace_unknown_words)
    preprocess_file(spath_test, spath_test_pp, replace_unknown_words, svocab)
    preprocess_file(tpath_test, tpath_test_pp, False)

    return spath_train_pp, tpath_train_pp, spath_test_pp, tpath_test_pp

def preprocess_file(path, path_to_preprocessed, replace_unknown_words, vocab = None):

    # tokenize
    language = path[-2:]
    os.system(
        f'./lib/tokenizer.perl -l {language} < {path} > {path_to_preprocessed}'
    )

    # lowercase
    text = lowercase(path_to_preprocessed)

    # split in sentences
    sentences = text.split('\n')

    # replace low frequency words in training data
    # or unknown words in test data (for source language only)
    if replace_unknown_words:
        sentences_tokenized = [s.split(' ') for s in sentences]
        if vocab: # test data
            sentences_tokenized = _replace_unknown_words(sentences_tokenized, vocab)
        else: # training data
            (sentences_tokenized, vocab) = _replace_low_frequency_words(sentences_tokenized)
        sentences = [' '.join(s) for s in sentences_tokenized]

    # write to file
    write_to_file(path_to_preprocessed, sentences)

    # Byte-pair encodings (BPE)
    # TODO

    return vocab


def postprocess(path):
    raise NotImplementedError()  

def lowercase(fname):
    f = open(fname, 'r')
    text = f.read()
    return text.lower().strip()

def _replace_low_frequency_words(sentences_tokenized):
    words = [w  for s in sentences_tokenized for w in s]
    word_counts = Counter(words)
    vocabulary = set(words)
    vocabulary_counts = Counter(vocabulary)
    frequent_word_counts = word_counts - vocabulary_counts
    vocabulary_frequent_words = list(frequent_word_counts.keys())
    if len(vocabulary_frequent_words) == len(vocabulary):
        raise('We just assume that infrequent words exist ...')
    vocabulary_frequent_words.append('<LOW>')
    sentences_frequent_words = _replace_unknown_words(sentences_tokenized, vocabulary_frequent_words)
    return (sentences_frequent_words, vocabulary_frequent_words)

def _replace_unknown_words(sentences_tokenized, vocabulary):
    replace_infrequent = lambda w: w if w in vocabulary else '<UNKNOWN>'
    return [[replace_infrequent(w) for w in s] for s in sentences_tokenized ]

def write_to_file(fpath, sentences):
    result_sentences = [f'{s}\n' for s in sentences]
    with open(fpath, 'w') as out:
        out.writelines(result_sentences)

def _path_to_preprocessed(path):
    return fp.path_to_outputfile(path, '.preprocessed')


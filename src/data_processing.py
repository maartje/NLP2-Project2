import filepaths as fp
import os
from collections import Counter

def preprocess(
        spath_train, tpath_train, spath_test, tpath_test,
        max_sentence_length = 50,
        replace_unknown_words = True, 
        apply_bpe = True, num_operations = 400, vocab_threshold = 5):

    # create output dir if not exists
    if not os.path.exists(fp.path_output_dir):
        os.makedirs(fp.path_output_dir)

    # paths to result files
    spath_train_pp = fp.path_to_outputfile(spath_train, '.preprocessed-train')
    tpath_train_pp = fp.path_to_outputfile(tpath_train, '.preprocessed-train')
    spath_test_pp = fp.path_to_outputfile(spath_test, '.preprocessed-test')
    tpath_test_pp = fp.path_to_outputfile(tpath_test, '.preprocessed-test')

    # TODO: use cache?
    # if not(useCache) or (not os.path.isfile(path_to_preprocessed)):
    # else:
    #     print (f'info: uses cached preprocessed file {path_to_preprocessed}')
 
    
    s_train_sentences = preprocess_file(spath_train, spath_train_pp)
    t_train_sentences = preprocess_file(tpath_train, tpath_train_pp)
    s_test_sentences = preprocess_file(spath_test, spath_test_pp)
    t_test_sentences = preprocess_file(tpath_test, tpath_test_pp)

    # filter on max sentence length
    (s_train_sentences, t_train_sentences) = filter_on_max_slength(
        s_train_sentences, t_train_sentences, max_sentence_length)
    (s_test_sentences, t_test_sentences) = filter_on_max_slength(
        s_test_sentences, t_test_sentences, max_sentence_length)

    if replace_unknown_words:
        (s_train_sentences, svocab) = replace_with_unknown(s_train_sentences, vocab = None)
        (t_train_sentences, _) = replace_with_unknown(t_train_sentences, vocab = None)
        (s_test_sentences, _) = replace_with_unknown(s_test_sentences, vocab = svocab)

    # write to file
    write_to_file(spath_train_pp, s_train_sentences)
    write_to_file(tpath_train_pp, t_train_sentences)
    write_to_file(spath_test_pp, s_test_sentences)
    write_to_file(tpath_test_pp, t_test_sentences)

    # Byte-pair encodings (BPE)
    if apply_bpe:
        path_codes = fp.path_to_outputfile(spath_train, '.BPE.codes')[:-3]
        spath_vocab = fp.path_to_outputfile(spath_train, '.BPE.vocab')
        tpath_vocab = fp.path_to_outputfile(tpath_train, '.BPE.vocab')
        spath_train_bpe = fp.path_to_outputfile(spath_train, '.bpe-train')
        tpath_train_bpe = fp.path_to_outputfile(tpath_train, '.bpe-train')
        spath_test_bpe = fp.path_to_outputfile(spath_test, '.bpe-test')
        BPE(
            spath_train_pp, tpath_train_pp, spath_test_pp,
            spath_train_bpe, tpath_train_bpe, spath_test_bpe,
            path_codes, spath_vocab, tpath_vocab,
            num_operations, vocab_threshold)

        # cleanup some temporary files
        os.remove(path_codes)
        os.remove(spath_vocab)
        os.remove(tpath_vocab)
        #os.remove(spath_train_pp)
        #os.remove(tpath_train_pp)
        #os.remove(spath_test_pp)
        return spath_train_bpe, tpath_train_bpe, spath_test_bpe, tpath_test_pp

    return spath_train_pp, tpath_train_pp, spath_test_pp, tpath_test_pp

def BPE(spath_train, tpath_train, spath_test,
        spath_train_out, tpath_train_out, spath_test_out,
        path_codes, spath_vocab, tpath_vocab,
        num_operations, vocab_threshold):
    # extract BPE vocablary and codes from concatenated training files
    os.system(f'subword-nmt learn-joint-bpe-and-vocab --input {spath_train} {tpath_train} -s {num_operations} -o {path_codes} --write-vocabulary {spath_vocab} {tpath_vocab}')

    # apply BPE on training files and on test file in the source language
    apply_bpe(spath_train, spath_train_out, path_codes, spath_vocab, vocab_threshold)
    apply_bpe(tpath_train, tpath_train_out, path_codes, tpath_vocab, vocab_threshold)
    apply_bpe(spath_test, spath_test_out, path_codes, spath_vocab, vocab_threshold)

def apply_bpe(path_input, path_output, path_codes, path_vocab, vocab_threshold):
    os.system(
        f'subword-nmt apply-bpe -c {path_codes} --vocabulary {path_vocab} --vocabulary-threshold {vocab_threshold} < {path_input} > {path_output}'
    )


def preprocess_file(path, path_to_preprocessed):

    # tokenize
    language = path[-2:]
    os.system(
        f'./lib/tokenizer.perl -l {language} < {path} > {path_to_preprocessed}'
    )

    # lowercase
    text = lowercase(path_to_preprocessed)

    # split in sentences
    sentences = text.split('\n')

    return sentences

def replace_with_unknown(sentences, vocab = None):

    # replace low frequency words in training data
    # or unknown words in test data (for source language only)
    sentences_tokenized = [s.split(' ') for s in sentences]
    if vocab: # test data
        sentences_tokenized = _replace_unknown_words(sentences_tokenized, vocab)
    else: # training data
        (sentences_tokenized, vocab) = _replace_low_frequency_words(sentences_tokenized)
    sentences = [' '.join(s) for s in sentences_tokenized]

    return sentences, vocab


def postprocess(path_input, path_output, undo_bpe = True):
    if not undo_bpe:
        return
    os.system(f"sed -r 's/(@@ )|(@@ ?$)//g' {path_input} > {path_output}")
 
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

def filter_on_max_slength(s_sentences, t_sentences, max_length):
    nr_of_lines = len(s_sentences)
    s_sentences_out = [
        s_sentences[i] for i in range(nr_of_lines) if len(s_sentences[i].split(' ')) <= max_length
    ]
    t_sentences_out = [
        t_sentences[i] for i in range(nr_of_lines) if len(s_sentences[i].split(' ')) <= max_length
    ]
    return (s_sentences_out, t_sentences_out)


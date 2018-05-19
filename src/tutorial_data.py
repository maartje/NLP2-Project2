import unicodedata
import re

def processData(max_length = 10, filter_on_prefix = True):
    fname = '../tutorial_data/eng-fra.txt'
    pairs = readSentencePairs(fname)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs, max_length, filter_on_prefix)
    print("Trimmed to %s sentence pairs" % len(pairs))
    english_sentences = [p[0] for p in pairs]
    french_sentences = [p[1] for p in pairs]
    fpath_en = '../tutorial_data/tutorial.en'
    fpath_fr = '../tutorial_data/tutorial.fr'
    write_to_file(fpath_en, english_sentences)
    write_to_file(fpath_fr, french_sentences)
    return (fpath_fr, fpath_en)

def readSentencePairs(fname):

    # Read the file and split into lines
    lines = open(fname, encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    return pairs


# Turn a Unicode string to plain ASCII
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

######################################################################
# Since there are a *lot* of example sentences and we want to train
# something quickly, we'll trim the data set to only relatively short and
# simple sentences. Here the maximum length is 10 words (that includes
# ending punctuation) and we're filtering to sentences that translate to
# the form "I am" or "He is" etc. (accounting for apostrophes replaced
# earlier).
#

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p, max_length, filter_on_prefix):
    return len(p[0].split(' ')) < max_length and \
        len(p[1].split(' ')) < max_length and \
        ((not filter_on_prefix) or p[0].startswith(eng_prefixes))


def filterPairs(pairs, max_length, filter_on_prefix):
    return [pair for pair in pairs if filterPair(pair, max_length, filter_on_prefix)]


def write_to_file(fpath, sentences):
    result_sentences = [f'{s}\n' for s in sentences]
    with open(fpath, 'w') as out:
        out.writelines(result_sentences)

#https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

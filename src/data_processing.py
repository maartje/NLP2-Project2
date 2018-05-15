import filepaths as fp
import os

def preprocess(path):
    path_to_preprocessed = fp.path_to_preprocessed(path)
    language = path[-2:]
    if not os.path.isfile(path_to_preprocessed):

        # tokenize
        os.system(
            f'./lib/tokenizer.perl -l {language} < {path} > {path_to_preprocessed}'
        )

        # lowercase
        lowercase(path_to_preprocessed)

        # Byte-pair encodings (BPE)
        # TODO
    return path_to_preprocessed


def postprocess(path):
    raise NotImplementedError()  

def lowercase(fname):
    f = open(fname, 'r')
    text = f.read()
    lines = [text.lower() for line in fname]
    with open(fname, 'w') as out:
        out.writelines(lines)

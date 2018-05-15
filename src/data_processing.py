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

	# fix errors in sentences
        fix_sentence_errors(path_to_preprocessed)

        # Byte-pair encodings (BPE)
        # TODO
    return path_to_preprocessed


def postprocess(path):
    raise NotImplementedError()  

def lowercase(fname):
    f = open(fname, 'r')
    text = f.read()
    with open(fname, 'w') as out:
        out.write(text.lower())

def fix_missing_dot(line):
    if line[-2] == '.':
        return(line)            
    else:
        return(line[:-2] + " ." + line[-1:])

def fix_sentence_errors(fpath):
    with open(fpath, 'r') as lines:
        fixed = [fix_missing_dot(line) for line in lines]
    with open(fpath, 'w') as out:
        out.writelines(fixed)
    return fixed


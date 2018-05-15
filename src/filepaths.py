from pathlib import Path

spath_train = "../train/train.fr" 
tpath_train = "../train/train.en"

spath_val = "../val/val.fr" 
tpath_val = "../val/val.en"

spath_test = "../test/test_2017_flickr.fr" 
tpath_test = "../test/test_2017_flickr.en"

def path_to_preprocessed(path):
    return _path_to_outputfile(path, '.preprocessed')

def path_to_postprocessed(path):
    return _path_to_outputfile(path, '.postprocessed')

def path_to_predicted(path):
    return _path_to_outputfile(path, '.predicted')

def _path_to_outputfile(path, addition):
    fpath = Path(path)
    fname = fpath.name[:-3] + addition + fpath.name[-3:]
    return str(Path('../output').joinpath(fname))



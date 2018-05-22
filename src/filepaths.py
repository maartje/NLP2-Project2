from pathlib import Path

spath_train = "../train/train.fr" 
tpath_train = "../train/train.en"

spath_val = "../val/val.fr" 
tpath_val = "../val/val.en"

spath_test = "../test/test_2017_flickr.fr" 
tpath_test = "../test/test_2017_flickr.en"

spath_toy = "../toy_data/toy.fr" 
tpath_toy = "../toy_data/toy.en"

spath_tutorial =  '../tutorial_data/tutorial.fr' 
tpath_tutorial = '../tutorial_data/tutorial.en'

path_output_dir = '../output'

def path_to_postprocessed(path):
    return path_to_outputfile(path, '.postprocessed')

def path_to_predicted(path):
    return path_to_outputfile(path, '.predicted')

def path_to_bleu(path):
    return path_to_outputfile(path, '.BLEU')[:-3]

def path_to_outputfile(path, addition):
    fpath = Path(path)
    fname = fpath.name[:-3] + addition + fpath.name[-3:]
    return str(Path(path_output_dir).joinpath(fname))



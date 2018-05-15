import data_preparation as dp

def train(spath, tpath):
    (slang, stensors) = dp.prepare_training_data(spath)
    (tlang, ttensors) = dp.prepare_training_data(tpath)
    raise NotImplementedError()

def evaluate(spath, tpath):
    raise NotImplementedError()


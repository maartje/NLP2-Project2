import os

def BLUE(candidate, reference, output):
    os.system('./lib/multi-bleu.perl ' + reference + ' < ' + candidate + ' > ' + output)

def Meteor(references, predictions):
    raise NotImplementedError()

def TER(references, predictions):
    raise NotImplementedError()
    


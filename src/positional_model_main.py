import data_preparation as dp
import filepaths as fp
import numpy as np
from torch.autograd import Variable

from rnn_model import EncoderRNN, AttnDecoderRNN
from rnn_model_train import trainIters
from rnn_model_predict import predict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
from torch.utils import data

def batch_gen(source, target, batch_size):
    ''' Batch generator  '''
    d = len(source) # amount of samples
    # for when batch size and data size are not compatible
    d = int((np.floor(d / batch_size)) * batch_size)
    for i in range(0, d, batch_size):
        x = source[i:i+batch_size]
        y = target[i:i+batch_size]
        yield Variable(x), Variable(y)

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=2) #padding_idx=2 ignores 2 index at training
    def forward(self, input):
        embedded = self.embedding(input)#.view(1, 1, -1)
        output = embedded
        return output

# parameters
emb_size = 128
max_length = 30
epochs = 1
batch_size = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

slang, tlang, index_array_pairs = dp.prepare_training_data(
    fp.spath_toy, fp.tpath_toy, max_length)
s_voc = slang.n_words
t_voc = tlang.n_words
source, target = zip(*index_array_pairs)
source = torch.from_numpy(np.array(source))
target = torch.from_numpy(np.array(target))
batch_data = batch_gen(source, target, batch_size)
epochs = 1
encoder = EncoderRNN(s_voc,emb_size, batch_size)
encoder_pos = EncoderRNN(max_length, emb_size, batch_size)
for e in range(epochs):
    for idx, batch in enumerate(batch_data):
        source, target = batch
        # forward pass
        word_embedding = torch.zeros(batch_size, max_length, emb_size)
        # loop over words
        pos_embedding = torch.zeros(batch_size, max_length, emb_size)
        for i in range(max_length):
            source_word = source[:,i]
            # positional encoding
            # just take ith index unless i==2(PAD token) then end of sentence is reached
            pos_ind = i
            if source_word[1] == 2:
                pos_ind = 2
            w_emb = encoder(source_word)
            word_embedding[:,i,:] = w_emb
            p_emb = encoder_pos(Variable(torch.from_numpy(np.array(pos_ind))))
            pos_embedding[:,i,:] = p_emb
        # Sum the embeddings can be seen as hidden state in our model
        hidden_state = word_embedding + pos_embedding
        print('idx', idx)
        print(hidden_state.size())

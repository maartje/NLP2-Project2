import filepaths as fp
import data_preparation as dp

import torch
from rnn_model import EncoderRNN, AttnDecoderRNN
from rnn_model_train import trainIters
from rnn_model_predict import predict

from plots import showLosses, showAttention

def train(spath, tpath, n_iters, hidden_size = 256, dropout_p = 0.1):
    # data structures for training
    (slang, tlang, index_array_pairs, max_length) = dp.prepare_training_data(spath, tpath)

    # create Encoder/Decoder models 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder1 = EncoderRNN(slang.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, tlang.n_words, max_length, dropout_p).to(device)

    # train and return losses for plotting
    plot_losses = trainIters(
        index_array_pairs, encoder1, attn_decoder1,  
        n_iters, max_length, print_every=n_iters/25., plot_every=n_iters/100.)

    # plot the losses
    showLosses(plot_losses, n_iters/100.)

    return encoder1, attn_decoder1, max_length



def evaluate(spath, tpath):
    raise NotImplementedError()


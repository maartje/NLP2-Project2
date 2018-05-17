import filepaths as fp
import data_preparation as dp

import torch
from rnn_model import EncoderRNN, AttnDecoderRNN
from rnn_model_train import trainIters
from rnn_model_predict import predict, predict_all

from plots import showLosses, showAttention
from data_processing import preprocess

from evaluate import BLUE 
import random

def train(spath, tpath, n_iters, hidden_size = 256, dropout_p = 0.1, useCache = True):
    # data structures for training
    (slang, tlang, index_array_pairs, max_length) = dp.prepare_training_data(spath, tpath, useCache)

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

    print (f'Losses diagram saved in TODO')
    print (f'Models saved in TODO')

    return encoder1, attn_decoder1, slang, tlang, max_length



def evaluate(spath, tpath, slang, tlang, 
    encoder1, attn_decoder1, max_length, useCache = True):

    # build source indices from test file 
    s_lists_of_indices = dp.prepare_test_data(slang, spath, useCache)
    
    # predict target indices
    (p_lists_of_indices, attentions) = predict_all(
        encoder1, attn_decoder1, s_lists_of_indices, max_length)
    
    # transform to target sentences (todo: post processing)
    p_lists_of_sentences = dp.sentenceFromIndexes_all(tlang, p_lists_of_indices)

    #TODO: post process: de-tokenize, true-case
    # HACK: tokenize and lowercase target sentences instead
    path_to_target = preprocess(tpath, useCache)

    # write predicted sentences to file
    path_to_predicted = fp.path_to_predicted(tpath)
    with open(path_to_predicted, 'w') as out:
        out.writelines(p_lists_of_sentences)

    print (f'Predictions written to {path_to_predicted}')

    # calculate evaluation scores
    path_to_BLUE = fp.path_to_bleu(tpath)
    evaluate.BLUE(path_to_predicted, path_to_target, path_to_BLUE)

    print (f'Blue score written to {path_to_BLUE}')

    # randomly plot some attention diagrams
    for i in range(3):
        r = random.randint(0, len(p_lists_of_indices))
        s_indices = s_lists_of_indices[r]
        p_indices = p_lists_of_indices[r]
        _attentions = attentions[r]
        s_words = dp.wordsFromIndexes(slang, s_indices)
        t_words = dp.wordsFromIndexes(tlang, p_indices)
        showAttention(s_words, t_words, _attentions)

    # randomly show some translations

    print (f'Attention diagrams saved in TODO')



    

    


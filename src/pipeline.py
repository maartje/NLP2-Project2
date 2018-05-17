import filepaths as fp
import data_preparation as dp

from plots import showLosses, showAttention
from data_processing import preprocess

from evaluate import BLUE 
import random

def run(spath_train, tpath_train, fn_train, 
        spath_test, tpath_test, fn_predict_all,
        max_length = 50, useCache = True):

    # data structures for training
    (slang, tlang, index_array_pairs) = dp.prepare_training_data(
        spath_train, tpath_train, max_length, useCache)

    # train and return losses for plotting
    (encoder, attn_decoder, plot_losses, plot_every) = fn_train(
        index_array_pairs, slang.n_words, tlang.n_words, max_length)

    # plot the losses
    showLosses(plot_losses, plot_every)

    print (f'Losses diagram saved in TODO')
    print (f'Models saved in TODO')
    print ()

    _evaluate(spath_test, tpath_test, slang, tlang, 
              encoder, attn_decoder, fn_predict_all,
              max_length, useCache = True)

    return encoder, attn_decoder, slang, tlang



def _evaluate(spath_test, tpath_test, slang, tlang, 
              encoder, attn_decoder, fn_predict_all,
              max_length, useCache = True):

    # build source indices from test file 
    s_lists_of_indices = dp.prepare_test_data(slang, spath_test, useCache)
    
    # predict target indices
    (p_lists_of_indices, attentions) = fn_predict_all(
        encoder, attn_decoder, s_lists_of_indices, max_length)
    
    # transform to target sentences (todo: post processing)
    p_lists_of_sentences = dp.sentenceFromIndexes_all(tlang, p_lists_of_indices)

    #TODO: post process: de-tokenize, true-case
    # HACK: tokenize and lowercase target sentences instead
    path_to_target = preprocess(tpath_test, useCache)

    # write predicted sentences to file
    path_to_predicted = fp.path_to_predicted(tpath_test)
    with open(path_to_predicted, 'w') as out:
        out.writelines(p_lists_of_sentences)

    print (f'Predictions written to {path_to_predicted}')

    # calculate evaluation scores
    path_to_BLUE = fp.path_to_bleu(tpath_test)
    BLUE(path_to_predicted, path_to_target, path_to_BLUE)

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



    

    


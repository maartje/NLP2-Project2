import filepaths as fp
import data_preparation as dp

from plots import showLosses, showAttention
from data_processing import preprocess, postprocess
from evaluation_wrapper import write_to_file

from evaluate import BLUE
import random
import torch

def run(spath_train, tpath_train,
        spath_test, tpath_test,
        fn_train, fn_predict_all,
        max_sentence_length = 50,
        replace_unknown_words = True,
        use_bpe = True, num_operations = 400, vocab_threshold = 5,
        padding = True):

    # data preprocessing
    (spath_train_pp, tpath_train_pp, spath_test_pp, tpath_test_pp) = preprocess(
        spath_train, tpath_train, spath_test, tpath_test,
        max_sentence_length,
        replace_unknown_words,
        use_bpe, num_operations, vocab_threshold)

    print (f'Data files preprocessed ...')
    print ()

    # data structures for training
    (slang, tlang, index_array_pairs, s_index_arrays_test, max_bpe_length) = dp.prepare_data(
        spath_train_pp, tpath_train_pp, spath_test_pp, padding)

    print (f'{len(index_array_pairs)} inputs constructed for training ...')
    print ()

    # train and return losses for plotting
    (encoder, attn_decoder, plot_losses, plot_every) = fn_train(
        index_array_pairs, slang.n_words, tlang.n_words, max_bpe_length)

    print (f'Training finished ...')
    print ()
    torch.save(encoder, 'encoder.pt')
    torch.save(attn_decoder, 'attn_decoder.pt')
    data = (s_index_arrays_test, slang, tlang, max_bpe_length)
    fname = 'data_run'
    with open(fname, "w") as file:
        file.write(str(data))
    # plot the losses
    showLosses(plot_losses, plot_every)

    print (f'Losses diagram saved in TODO')
    print (f'Models saved in TODO')
    print ()

    _evaluate(s_index_arrays_test, tpath_test_pp, slang, tlang,
              encoder, attn_decoder, fn_predict_all,
              max_bpe_length, use_bpe)

    return encoder, attn_decoder, slang, tlang, plot_losses, max_bpe_length



def _evaluate(s_lists_of_indices, tpath_test, slang, tlang,
              encoder, attn_decoder, fn_predict_all,
              max_bpe_length, use_bpe):

    print (f'{len(s_lists_of_indices)} inputs constructed for testing ...')
    print ()

    # predict target indices
    (p_lists_of_indices, attentions) = fn_predict_all(
        encoder, attn_decoder, s_lists_of_indices, max_bpe_length)

    print (f'{len(p_lists_of_indices)} outputs predicted ...')
    print ()

    # transform to target sentences (todo: post processing)
    p_lists_of_sentences = dp.sentenceFromIndexes_all(tlang, p_lists_of_indices)

    # tokenize and lowercase target sentences for comparison
    path_to_target = tpath_test

    # write predictions to file
    path_to_predicted = fp.path_to_outputfile(tpath_test, '.predicted')
    write_to_file(path_to_predicted, p_lists_of_sentences)
    # with open(path_to_predicted, 'w') as out:
    #     out.writelines(p_lists_of_sentences)
    #     out.write('\n')

    # undo BPE encoding for predictions
    path_to_postprocessed = path_to_predicted
    if use_bpe:
        path_to_postprocessed = fp.path_to_outputfile(path_to_predicted, '.postprocessed')
        postprocess(path_to_predicted, path_to_postprocessed, use_bpe)

    print (f'Predictions written to {path_to_postprocessed}')

    # calculate evaluation scores
    path_to_BLUE = fp.path_to_bleu(tpath_test)
    BLUE(path_to_postprocessed, path_to_target, path_to_BLUE)

    print (f'Blue score written to {path_to_BLUE}')

    # randomly plot some attention diagrams
    for i in range(3):
        r = random.randint(0, len(p_lists_of_indices)-1)
        s_indices = s_lists_of_indices[r]
        p_indices = p_lists_of_indices[r]
        _attentions = attentions[r]
        s_words = dp.wordsFromIndexes(slang, s_indices)
        p_words = dp.wordsFromIndexes(tlang, p_indices)
        showAttention(s_words, p_words, _attentions.numpy(), use_bpe)

    # randomly show some translations

    print (f'Attention diagrams saved in TODO')

import data_preparation as dp
import filepaths as fp

from rnn_model import EncoderRNN, AttnDecoderRNN
from rnn_model_train import trainIters
from rnn_model_predict import predict

import torch

hidden_size = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

(slang, tlang, tensor_pairs) = dp.prepare_training_data(
    fp.spath_toy, fp.tpath_toy)
MAX_LENGTH = 10 # maximum sentence length
#TODO filter sentences
#TODO max length property in lang class?
#TODO: prediction: in/out vector or sentence?

encoder1 = EncoderRNN(slang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, tlang.n_words, MAX_LENGTH, dropout_p=0.1).to(device)

trainIters(tensor_pairs, encoder1, attn_decoder1, MAX_LENGTH, 75000, print_every=5000)

predictRandomly(pairs, encoder1, attn_decoder1, MAX_LENGTH)

def predictAndShowAttention(s_sentence, max_length):
    s_tensor = dp.tensorFromSentence(input_lang, s_sentence)
    output_words, attentions = predict(
        encoder1, attn_decoder1, s_tensor, max_length)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)

def predictRandomly(pairs, encoder, decoder, max_length, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = predict(encoder, decoder, pair[0], max_length)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

predictAndShowAttention("elle a cinq ans de moins que moi .", MAX_LENGTH)

predictAndShowAttention("elle est trop petit .", MAX_LENGTH)

predictAndShowAttention("je ne crains pas de mourir .", MAX_LENGTH)

predictAndShowAttention("c est un jeune directeur plein de talent .", MAX_LENGTH)




import pipeline
import filepaths as fp
import torch

#from rnn_model import EncoderRNN, AttnDecoderRNN
#from rnn_model_train import trainIters
#from rnn_model_predict import predict_all

from pos_model import EncoderPOS, AttnDecoderPOS
from pos_model_train import trainIters as trainItersPOS
from pos_model_predict import predict_all as predict_allPOS

def train_model_pos(index_array_pairs, s_vocab_size, t_vocab_size, max_length):
    
    # create Encoder/Decoder models 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = EncoderPOS(s_vocab_size, max_length, hidden_size).to(device)
    attn_decoder = AttnDecoderPOS(hidden_size, t_vocab_size, max_length, dropout_p).to(device)

    # train models and return losses to plot
    plot_losses = trainItersPOS(
        index_array_pairs, encoder, attn_decoder, n_epochs, max_length, 
        print_every, plot_every = plot_every, 
        learning_rate = learning_rate, max_hours = max_hours, clip=clip)
    
    # return trained models and info to plot the losses
    return encoder, attn_decoder, plot_losses, plot_every


#### TRAIN and TEST DATA WITHOUT BPE
# verplaats directory van je modelsss na runnen 
hidden_size = 256
dropout_p = 0.1
learning_rate = 0.01
n_epochs = 20
max_hours = 23
clip = 8
use_bpe = False
replace_unknown_words = True
padding = True
MAX_LENGTH = 25
plot_every = 200
print_every = 28319 # every epoch

encoder, attn_decoder, slang, tlang, plot_losses, max_bpe_length = pipeline.run(
    fp.spath_train, fp.tpath_train, 
    fp.spath_test, fp.tpath_test, 
    train_model_pos, predict_allPOS, 
    max_sentence_length = MAX_LENGTH, 
    replace_unknown_words = True, 
    use_bpe = False, num_operations = 100, vocab_threshold = 1, 
    padding = False, model_name = 'pos_no_bpe')

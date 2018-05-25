import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1

def calculate_loss(encoder, decoder, input_tensor, target_tensor, max_length):
    with torch.no_grad():
        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden


        criterion = nn.NLLLoss()
        prediction_length = 0
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            loss += criterion(decoder_output, target_tensor[di])
            prediction_length += 1
            if topi.item() == EOS_token:
                break

    return loss.item() / prediction_length 
    #  divide by prediction (instead of target) length to avoid rewarding short predictions

def calculate_average_loss(encoder, decoder, tensor_pairs, max_length):
    return np.mean([calculate_loss(encoder, decoder, p[0], p[1], max_length) for p in tensor_pairs])



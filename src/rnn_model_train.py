import torch
import torch.nn as nn
from torch import optim
import random
import time
from debug import timeSince
from datetime import datetime, timedelta
from torch.nn.utils import clip_grad_norm_
from itertools import chain

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

teacher_forcing_ratio = 2 #0.5

SOS_token = 0
EOS_token = 1

def train(input_tensor, target_tensor, encoder, decoder, 
        optimizer, criterion, max_length, clip):
    encoder_hidden = encoder.initHidden()

    optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    if clip:
        clip_grad_norm_(decoder.parameters(), clip) # MJ: to prevent peaks
        clip_grad_norm_(encoder.parameters(), clip) # MJ: to prevent peaks

    optimizer.step()

    return loss.item() / target_length


def trainIters(index_array_pairs, encoder, decoder, n_epochs, max_length,
        print_every=1000, plot_every=100, learning_rate=0.01, max_hours = 24, clip = None):
    start = time.time()
    start_time = datetime.now()
    end_time = start_time + timedelta(hours = max_hours)


    tensor_pairs = [
        (
            torch.tensor(s_indices, dtype=torch.long, device=device).view(-1, 1),
            torch.tensor(t_indices, dtype=torch.long, device=device).view(-1, 1),
        )
        for (s_indices, t_indices) in index_array_pairs
    ]

    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    optimizer = optim.SGD(chain(encoder.parameters(), decoder.parameters()), lr=learning_rate)
    criterion = nn.NLLLoss()
    loss_peak = False
    
    n_iters = n_epochs * len(tensor_pairs)
    for epoch in range(n_epochs):

        for i in range(0, len(tensor_pairs)):
            iter = i + 1 + epoch * len(tensor_pairs)
            training_pair = tensor_pairs[i]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            loss = train(input_tensor, target_tensor, encoder,
                         decoder, optimizer, criterion, max_length, clip)

            # MJ: debug peaks in loss diagram
            if not loss_peak and plot_losses and (loss > 2.5 * plot_losses[-1]) : 
                loss_peak = True
                print ('peak in loss diagram: ')
                print (' current loss', loss)
                print ('avg prev loss', plot_losses[-1])
                print ('epoch', epoch)
                print ('sentence index', i)
                print ()

            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                             iter, iter / n_iters * 100, print_loss_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
                loss_peak = False

            if datetime.now() > end_time:
                print(f'exceeded max hours {max_hours}')
                break

    return plot_losses




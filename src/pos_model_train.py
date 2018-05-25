import torch
import torch.nn as nn
from torch.autograd import Variable

from torch import optim
import random
import time
from debug import timeSince
from datetime import datetime, timedelta
from torch.nn.utils import clip_grad_norm_



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

teacher_forcing_ratio = 2

SOS_token = 0
EOS_token = 1

def train(input_tensor, target_tensor, encoder, decoder,
        encoder_optimizer, decoder_optimizer, criterion, max_length, clip):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    # decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0
    for ei in range(input_length):
        pos_tensor = torch.tensor([ei], device=device)
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], pos_tensor, encoder_hidden)
        hidden_state = encoder_output + encoder_hidden
        encoder_outputs[ei] = hidden_state[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_outputs.mean(0).view(1,1,encoder.hidden_size)

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
    model_param = list(decoder.parameters()) + list(encoder.parameters())
    if clip:
        clip_grad_norm_(model_param, clip)
    encoder_optimizer.step()
    # decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(index_array_pairs, encoder, decoder, n_epochs, max_length,
        print_every=1000, plot_every=100, learning_rate=0.01, max_hours = 24, clip=10):
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
    parameters = list(encoder.parameters()) + list(decoder.parameters())
    encoder_optimizer = optim.SGD(parameters, lr=learning_rate)
    # encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    # decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    decoder_optimizer = []
    # training_pairs = [random.choice(tensor_pairs) for i in range(n_iters)]
    # training_pairs = tensor_pairs
    criterion = nn.NLLLoss()
    size_data = len(tensor_pairs)
    n_iters = n_epochs * size_data
    for epoch in range(n_epochs):

        for i in range(0, size_data):
            iter = i + 1 + epoch * size_data
            training_pair = tensor_pairs[i]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            loss = train(input_tensor, target_tensor, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion, max_length, clip)
            print_loss_total += loss
            plot_loss_total += loss
            # print(iter, print_every)
            # print(iter % print_every)
            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                             iter, iter / n_iters * 100, print_loss_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

            if datetime.now() > end_time:
                print(f'exceeded max hours {max_hours}')
                break

    return plot_losses

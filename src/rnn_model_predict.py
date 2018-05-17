import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1

def predict(encoder, decoder, s_indices, max_length):
    input_tensor = torch.tensor(s_indices, dtype=torch.long, device=device).view(-1, 1)
    with torch.no_grad():
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_indices = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            decoded_indices.append(topi.item())
            if topi.item() == EOS_token:
                break

            decoder_input = topi.squeeze().detach()

        return decoded_indices, decoder_attentions[:di + 1]

def predict_all(encoder, decoder, s_indices_list, max_length):
    return [predict(encoder, decoder, s_indices, max_length) for s_indices in s_indices_list]


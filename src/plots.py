import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showLosses(plot_losses, plot_every):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    iter_numbers = [i*plot_every for i in range(len(plot_losses))]
    plt.plot(iter_numbers, plot_losses)
    plt.show()


def showAttention(input_words, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_words, rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

    input_sentence = ' '.join(input_words)
    output_sentence = ' '.join(output_words)
    print (input_sentence)
    print (input_sentence.replace('@@ ', ''))
    print (output_sentence)
    print (output_sentence.replace('@@ ', ''))




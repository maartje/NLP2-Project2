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


def showAttention(input_words, output_words, attentions, merge_subwords = False):
    input_sentence = ' '.join(input_words)
    output_sentence = ' '.join(output_words)
    print (input_sentence)
    print (input_sentence.replace('@@ ', ''))
    print (output_sentence)
    print (output_sentence.replace('@@ ', ''))

    if merge_subwords:
        (input_words, output_words, attentions) = merge_bpe(input_words, output_words, attentions)

    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions, cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_words, rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def merge_bpe(s_words, t_words, A):
    s_words_merged, X = merge_bpe_s(s_words, A)
    t_words_merged, attentions = merge_bpe_t(t_words, X)
    return (s_words_merged, t_words_merged, attentions)

def merge_bpe_s(s_words, A):
    merge_indices = [i for i, w in enumerate(s_words) if w.endswith('@@')]
    resulting_columns = []
    merge_column = np.array([])
    resulting_words = []
    merge_word = ''
    for i, column in enumerate(A.T):
        if not merge_column.any():
            merge_column = column
            merge_word = s_words[i] if i < len(s_words) else ''
        else:
            merge_column = (merge_column + column)
            merge_word += s_words[i]
        if i not in merge_indices:
            resulting_words.append(merge_word)
            resulting_columns.append(merge_column)
            merge_column = np.array([])
    return [w.replace('@@', '') for w in resulting_words], np.column_stack(resulting_columns)
        
def merge_bpe_t(t_words, X):
    merge_indices = [i for i, w in enumerate(t_words) if w.endswith('@@')]
    resulting_rows = []
    merge_row = np.array([])
    resulting_words = []
    merge_word = ''
    for i, row in enumerate(X):
        if not merge_row.any():
            merge_row = row
            merge_word = t_words[i] if i < len(t_words) else ''
        else:
            merge_row = (merge_row + row)/2.
            merge_word += t_words[i]
        if i not in merge_indices:
            resulting_words.append(merge_word)
            resulting_rows.append(merge_row)
            merge_row = np.array([])
    return [w.replace('@@', '') for w in resulting_words], np.row_stack(resulting_rows)
        



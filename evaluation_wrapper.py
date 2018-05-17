import os

def write_to_file(file_name, sentences, trans=False):
    ''' Write list of sentences to a file. Optional to write into
    TRANS format.

    file_name: string of file name
    sentences: list of sentences ['sent1', 'sent2']
    trans: flag to write into TRANS format
    '''
    with open(file_name, 'w') as file:
        for i, sent in enumerate(sentences):
            file.write(sent)
            if trans:
                file.write(' (' + str(i) + ')')
            file.write('\n')

# assumes evaluation methods in the same directory
def bleu_score(candidate, reference, output):
    os.system('./multi-bleu.perl ' + reference + ' < ' + candidate + ' > ' + output)

def meteor_score(candidate, reference, output):
    path_meteor = os.getcwd() + '/meteor-1.5/'
    os.system('java -Xmx2G -jar '+ path_meteor + 'meteor*.jar ' + candidate + ' ' + reference + ' > ' + output)

def ter_score(candidate, reference, output):
    path_tercom = os.getcwd() + '/tercom-0.7.25/'
    # os.system('java -jar ' + path_tercom + 'tercom.7.25.jar -r ' + reference + ' -h ' + candidate + ' -n ' + output + ' > ' + output)
    os.system('java -jar ' + path_tercom + 'tercom.7.25.jar -r ' + reference + ' -h ' + candidate + ' > ' + output)

def main():
    # example toy data usage
    candidate = ["this is a test", "this is not a test"]
    reference = ["this is a test", "this is test"]
    write_to_file('candidate_trans.txt', candidate, True)
    write_to_file('reference_trans.txt', reference, True)
    write_to_file('candidate.txt', candidate)
    write_to_file('reference.txt', reference)
    ter_score('candidate_trans.txt', 'reference_trans.txt', 'ter_scores.txt')
    bleu_score('candidate.txt', 'reference.txt', 'bleu_scores.txt')
    meteor_score('candidate.txt', 'reference.txt', 'meteor_score.txt')

if __name__ == "__main__":
    main()

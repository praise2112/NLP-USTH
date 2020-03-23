from collections import defaultdict


def bigram_add_one():
    counts = defaultdict(int)
    context_counts = defaultdict(int)
    with open("Iamsam2.txt") as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            words = line.split()

            for i in range(0, len(words)):
                two_word_join = words[i] + " " + words[i-1]
                counts[two_word_join] += 1
                context_counts[words[i]] += 0.5
                pass

    for ngram, count in counts.items():
        print(ngram + "\t" + "{}".format((counts[ngram]+1)/(context_counts[ngram.split()[0]] + len(context_counts))))


from collections import defaultdict


def train_bigram(train_file, model_file):
    """Train trigram language model and save to model file
    """
    counts = defaultdict(int)  # count the n-gram
    context_counts = defaultdict(int)  # count the context
    with open(train_file) as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            words = line.split()

            for i in range(0, len(words)):
                counts[str(words[i]) + " " + str(words[i-1])] += 1  # Add bigram and bigram context
                context_counts[str(words[i])] += 0.5
                pass

    # Save probabilities to the model file
    with open(model_file, 'w') as fo:
        for ngram, count in counts.items():
            context = ngram.split()
            context.pop()
            context = "".join(context)
            probability = (counts[ngram] +1 ) / (context_counts[context] + len(context_counts))
            fo.write('%s\t%f\n' % (ngram, probability))
            pass



train_bigram('Iamsam2.txt', 'model.txt')



def trigram_train():
    counts = defaultdict(int)
    context_counts = defaultdict(int)
    with open("Iamsam.txt") as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            words = line.split()

            for i in range(2, len(words)):
                two_word_join = words[i - 2] + " " + words[i - 1]
                three_word_join = words[i - 2] + " " + words[i - 1] + " " + words[i]

                counts[three_word_join] += 1
                context_counts[two_word_join] += 1
                pass

    print(counts)
    print(context_counts)
    for ngram, count in counts.items():
        context = ngram.split()[0:2]
        print(ngram + "\t" + "{}".format(counts[ngram] / context_counts[" ".join(context)]) + "\n")


import math
def test_bigram(test_file, lambda1=0.95, N=1000000):
    W = 0  # Total word count
    H = 0
    probs = {'0': 91/100, '1': 1/100, '2': 1/100, '3': 1/100, '4': 1/100, '5': 1/100, '6': 1/100, '7': 1/100, '8': 1/100, '9': 1/100}
    with open(test_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            words = line.split()

            for i in range(0, len(words)):
                p1 = None

                p1 = float(lambda1) * float(probs.get(str(words[i]), 0)) + float(1 - lambda1) / float(N)

                W += 1  # Count the words
                H += -math.log2(p1)  # We use logarithm to avoid underflow
    H = H / W
    P = 2 ** H

    print("Entropy: {}".format(H))
    print("Perplexity: {}".format(P))

    return P


test_bigram('test.txt')
# bigram_add_one()
# trigram_train()
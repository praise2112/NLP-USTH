# Naive Bayes for Sentiment Classification

# Loading Data
import re


def load_data(file_path):
    data = []
    # Regular expression to get the label and the text
    regx = re.compile(r'^(\+1|-1)\s+(.+)$')
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            match = regx.match(line)
            if match:
                lb = match.group(1)
                text = match.group(2)
                data.append((text, lb))
    return data


data = load_data('./sentiment.txt')

print(data[0])
print(data[-1])



# Train/test split
from sklearn.model_selection import train_test_split

texts, labels = zip(*data)
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

from collections import Counter

print(Counter(train_labels))
print(Counter(test_labels))

import string


# Training Multinomial Naive Bayes Model
def build_vocab(texts):
    """Build vocabulary from dataset

    Args:
        texts (list): list of tokenized sentences

    Returns:
        vocab (dict): map from word to index
    """
    vocab = {}
    for s in texts:
        for word in s.split():
            # Check if word is a punctuation
            if word in string.punctuation:
                continue
            if word not in vocab:
                idx = len(vocab)
                vocab[word] = idx
    return vocab


vocab = build_vocab(train_texts)
print(vocab)

from collections import defaultdict
import math


def train_naive_bayes(texts, labels, target_classes, alpha=1):
    """Train a multinomial Naive Bayes model
    """
    ndoc = 0
    nc = defaultdict(int)  # map from a class label to number of documents in the class
    logprior = dict()
    loglikelihood = dict()
    count = defaultdict(int)  # count the occurrences of w in documents of class c

    vocab = build_vocab(texts)
    # Training
    for s, c in zip(texts, labels):
        ndoc += 1
        nc[c] += 1

        # print("Beee  "+s)
        # sentence_ = negateText(s)  # negating the text by , prepending the prefix NOT to every word after a token of logical negation (n’t, not, no, never) until the next punctuation mark\
        # print("Wooo  "+sentence_)
        # s_without_duplicate = list(dict.fromkeys(sentence_))  # removing duplicates from list
        # s_without_duplicate = list(dict.fromkeys(s.split()))    # removing duplicates from list
        for w in s.split():  # use w in s.split():  for multinomial, use s_without_duplicate for binarized
            if w in vocab:
                count[(w, c)] += 1

    print(count)

    vocab_size = len(vocab)
    for c in target_classes:
        logprior[c] = math.log(nc[c] / ndoc)
        sum_ = 0
        for w in vocab.keys():
            if (w, c) not in count: count[(w, c)] = 0
            sum_ += count[(w, c)]

        for w in vocab.keys():
            loglikelihood[(w, c)] = math.log((count[(w, c)] + alpha) / (sum_ + alpha * vocab_size))

    return logprior, loglikelihood, vocab


def negateText(sentence):
    import re
    # up to punctuation as in punct, put tags for words
    # following a negative word
    # find punctuation in the sentence
    # print("luuuuu  ")
    # print(re.findall(r'[.:;!?]', sentence))
    try:
        punct = re.findall(r'[.:;!?]', sentence)[0]
    except IndexError:
        punct = ''
    # create word set from sentence
    wordSet = {x for x in re.split("[.:;!?, ]", sentence) if x}
    keywordSet = {"don't", "never", "nothing", "nowhere", "noone", "none", "not",
                  "hasn't", "hadn't", "can't", "couldn't", "shouldn't", "won't",
                  "wouldn't", "don't", "doesn't", "didn't", "isn't", "aren't", "ain't"}
    # find negative words in sentence
    neg_words = wordSet & keywordSet
    if neg_words:
        for word in neg_words:
            start_to_w = sentence[:sentence.find(word) + len(word)]
            # put tags to words after the negative word
            w_to_punct = re.sub(r'\b([A-Za-z\']+)\b', r'NOT_\1',
                                sentence[sentence.find(word) + len(word):sentence.find(punct)])
            punct_to_end = sentence[sentence.find(punct):]
            # print(start_to_w + w_to_punct + punct_to_end)
            return start_to_w + w_to_punct + punct_to_end
    else:
        return sentence
        # print("no negative words found ...")


# TEST
# data = [
#     ("Chinese Beijing Chinese", "c"),
#     ("Chinese Chinese Shanghai", "c"),
#     ("Chinese Macao", "c"),
#     ("Tokyo Japan Chinese", "j")
# ]
# texts, labels = zip(*data)
# target_classes = ["c", "j"]
#
# logprior, loglikelihood, vocab = train_naive_bayes(texts, labels, target_classes)
# data = [
#     ("good good good great great great", "+"),
#     ("poor great great", "+"),
#     ("good poor poor poor", "-"),
#     ("good poor  poor poor poor poor great great", "-"),
#     ("poor poor", "-")
# ]
# texts, labels = zip(*data)
# target_classes = ["+", "-"]
#
# logprior, loglikelihood, vocab = train_naive_bayes(texts, labels, target_classes)


# confirmation for multinomial
# assert logprior['c'] == math.log(0.75)
# assert logprior['j'] == math.log(0.25)
# assert loglikelihood[('Chinese', 'c')] == math.log(3/7)
# assert loglikelihood[('Tokyo', 'c')] == math.log(1/14)
# assert loglikelihood[('Japan', 'c')] == math.log(1/14)
# assert loglikelihood[('Tokyo', 'j')] == math.log(2/9)

# confirmation for binarized
# print(loglikelihood[('Chinese', 'c')])
# print(10 ** loglikelihood[('Chinese', 'c')])
# print(math.log(math.exp(loglikelihood[('Chinese', 'c')])))
# assert logprior['c'] == math.log(0.75)
# assert logprior['j'] == math.log(0.25)
# assert loglikelihood[('Chinese', 'c')] == math.log(4/12)
# assert loglikelihood[('Beijing', 'c')] == math.log(2/12)


# Prediction Function
def test_naive_bayes(testdoc, logprior, loglikelihood, target_classes, vocab):
    sum_ = {}
    # print(logprior)
    # print(10 ** logprior['c'])
    # print(10 ** logprior['j'])

    for c in target_classes:
        sum_[c] = logprior[c]
        for w in testdoc.split():
            if w in vocab:
                sum_[c] += loglikelihood[(w, c)]
    print("www")
    print(sum_)
    # print(10 ** sum_['c'])
    # print(10 ** sum_['j'])
    # sort keys in sum_ by value
    sorted_keys = sorted(sum_.keys(), key=lambda x: sum_[x], reverse=True)
    return sorted_keys[0]


# TEST
# print('Predicted class: %s' % test_naive_bayes('good good great poor', logprior, loglikelihood, target_classes, vocab))

# multinomial predict c
# Binarized predicts j

# Now, it is time to train our Naive Bayes model on the sentiment data.
target_classes = ['+1', '-1']  # we can construct a fixed set of classes from train_labels
logprior, loglikelihood, vocab = train_naive_bayes(train_texts, train_labels, target_classes)

print('Predicted class: %s' % test_naive_bayes("enigma is well-made , but it's just too dry and too placid .", logprior,
                                               loglikelihood, target_classes, vocab))

# Evaluation
predicted_labels = [test_naive_bayes(s, logprior, loglikelihood, target_classes, vocab)
                    for s in test_texts]
from sklearn import metrics

print('Accuracy score: %f' % metrics.accuracy_score(test_labels, predicted_labels))

# We can calculate precision, recall, f1_score per class.

for c in target_classes:
    print('Evaluation measures for class %s' % c)
    print('  Precision: %f' % metrics.precision_score(test_labels, predicted_labels, pos_label=c))
    print('  Recall: %f' % metrics.recall_score(test_labels, predicted_labels, pos_label=c))
    print('  F1: %f' % metrics.f1_score(test_labels, predicted_labels, pos_label=c))

# We can also compute macro-averaged and micro-averaged f1 score.
print('Macro-averaged f1: %f' % metrics.f1_score(test_labels, predicted_labels, average='macro'))
print('Micro-averaged f1: %f' % metrics.f1_score(test_labels, predicted_labels, average='micro'))
#
# # We can report classification results all by once.
print(metrics.classification_report(test_labels, predicted_labels))

# Accuracy with binarized : 0.763244
# Accuracy with binarized and negation: 0.766057

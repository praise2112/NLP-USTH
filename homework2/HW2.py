
# Exercise 1
tagset = set()
with open('train.txt', encoding="utf8") as f:
    for line in f:
        line = line.rstrip()
        wordtags = line.split()
        for wordtag in wordtags:
            fields = wordtag.split('/')
            tag = fields[-1]
            if tag == 'chưa':
                print(wordtag)
            tagset.add(tag)

print('# Number of tags: %d' % len(tagset))


# Exercise 2: Training First-Order Hidden Markov Models for POS Tagging
from collections import defaultdict

def split_wordtag(wordtag):
    fields = wordtag.split('/')
    tag = fields.pop()
    word = '/'.join(fields)
    return word, tag


def train(train_file: str, model_file: str):
    emit = defaultdict(int)  # dictionary to store emission count C(t_i, w_i)
    transition = defaultdict(int)  # transition count C(t_{i-1}, t_i)
    context = defaultdict(int)  # count the context
    with open(train_file, 'r', encoding="utf8") as f:
        for line in f:
            line = line.strip()
            previous = '<s>'  # Make the sentence start
            context[previous] += 1
            wordtags = line.split()
            for wordtag in wordtags:
                word, tag = split_wordtag(wordtag)

                # TODO: Write code to update transition count, emission count
                # YOUR CODE HERE
                transition[str(previous + " " + tag)] += 1   # Count the transition
                context[tag] += 1  # Count the context
                emit[str(tag + " " + word)] += 1  # Count the emission
                previous = tag

                # END OF YOUR CODE

            transition[previous + ' </s>'] += 1  # Sentence stop

    # Now we will save the parameters of the model to a file
    with open(model_file, 'w', encoding="utf8") as fo:

        # Save transition probabilities
        for key, value in transition.items():
            previous, word = key.split(' ')
            fo.write('T %s %f\n' % (key, value / context[previous]))

        # TODO: Write emission probabilities into the model file
        # YOUR CODE HERE
        for key, value in emit.items():
            previous, word = key.split(' ')
            fo.write('E %s %f\n' % (key, value / context[previous]))

        # Save emission probabilities

        # END OF YOUR CODE
    print('Finished training first-order HMM!')


train('train.txt', 'HMM_model.txt')


# Exercise 3: Loading HMM model from file
def load_model(model_file: str):
    """Load saved HMM model
    """
    transition = defaultdict(lambda: 0)
    emission = defaultdict(lambda: 0)
    possible_tags = {}

    # TODO: Write your code to load the model file
    # YOUR CODE HERE
    with open(model_file, 'r', encoding="utf8") as f:
        for line in f:
            line = line.strip()
            type_, context, word, prob = line.split(' ')
            possible_tags[context] = 1  # We use this to enumerate all tags
            if type_ == "T":
                transition[str(context + " " + word)] = prob
            else:
                emission[str(context + " " + word)] = prob

    # END OF YOUR CODE HERE
    return transition, emission, possible_tags


transition, emission, possible_tags = load_model('HMM_model.txt')
print(list(possible_tags.keys()))
print(transition)
print(emission)


# Exercise 4: Viterbi algorithm
import math


def viterbi(line, transition, emission, possible_tags):
    """Infer the tag sequence for a tokenized sentence

    Args:
        line (str): a tokenized word sequence
                    e.g., "Chiều cuối thu , trời vùng_biển Nghi_Xuân ảm_đạm ."
        transition (dict): transition probabilities
        emission (dict): emission probabilities
    """
    words = line.split()
    l = len(words)
    best_score = {}
    best_edge = {}
    best_score[('0 <s>')] = 0  # Start with <s>
    best_edge[('0 <s>')] = None
    # Forward Step
    for i in range(l):
        for prev in possible_tags.keys():
            for _next in possible_tags.keys():
                if str(i) + ' ' + prev in best_score and prev + ' ' + _next in transition:
                    if emission[_next + ' ' + words[i]] == 0:
                        # To avoid zero probabilities, we use very small value
                        emission[_next + " " + words[i]] = 10 ** (-10)

                    # TODO: Write code to calculate the score of the path that
                    # connect the best path in the step i with the edge prev -> _next
                    print("it is")
                    print(best_score.get(str(i) + " " + str(prev), 0))
                    print(transition.get(str(prev + ' ' + _next), 0))
                    print(-math.log(float(transition.get(str(prev + ' ' + _next), 0))))
                    print(-math.log(float(emission.get(str(_next + ' ' + words[i]), 0))))
                    score = best_score.get(str(i) + " " + str(prev), 0) + -math.log(float(transition.get(str(prev + ' ' + _next), 0))) + -math.log(float(emission.get(str(_next + " " + words[i]), 0)))

                    # END OF YOUR CODE HERE
                    if str(i + 1) + " " + _next not in best_score or best_score[str(i + 1) + " " + _next] > score:
                        best_score[str(i + 1) + " " + _next] = score
                        best_edge[str(i + 1) + " " + _next] = str(i) + " " + prev

    for prev in possible_tags.keys():
        if str(l) + ' ' + prev in best_score:
            if (prev + ' ' + '</s>') not in transition:
                transition[prev + ' ' + '</s>'] = 10 ** (-10)

            # TODO: Calculate best_score[str(l+1) + ' </s>'] and best_edge[str(l+1) + ' </s>']
            # for the sentence top symbole '</s>'
            # The different from the other time step is that, we do not use emission probility in calculating score
            # YOUR CODE HERE
            print("it is--------")
            print(best_score.get(str(i) + " " + str(prev), 0))
            print(str(prev + ' ' + _next))
            print(transition.get(str(prev + ' ' + _next), 0))
            # print(-math.log(float(transition.get(str(prev + ' ' + _next), 0))))
            # lg = 0
            # try:
            lg = -math.log(float(transition.get(str(prev + ' ' + '</s>'), 0)))
            score = best_score.get(str(str(l) + ' ' + prev), 0) + lg
            # except ValueError:x
            #     lg = 0
            #     score = 0

            # if str(i + 1) + " " + _next not in best_score or best_score[str(i + 1) + " " + _next] > score:
            if str(l + 1) + " " + '</s>' not in best_score or best_score[str(l + 1) + " " + '</s>'] > score:
                best_score[str(l + 1) + " " + "</s>"] = score
                best_edge[str(l + 1) + " " + "</s>"] = str(l) + ' ' + prev
            # END OF YOUR CODE

    # Backward Step
    tags = []
    next_edge = best_edge[str(l + 1) + " " + "</s>"]
    # TODO: Complete the backward step in Viterbi algorithm
    # Finish the while loop in the pseudo code
    while next_edge != "0 <s>":
        # Add the substring for this edge to the words
        position, tag = next_edge.split(' ')
        tags.append(tag)
        next_edge = best_edge[next_edge]

    # END OF YOUR CODE
    tags.reverse()
    return ' '.join(tags)


tagSeq = viterbi('Chiều cuối thu , trời vùng_biển Nghi_Xuân ảm_đạm .', transition, emission, possible_tags)
print(tagSeq)



def evaluate(test_file: str, transition, emission, possible_tags):
    correct = 0
    total = 0

    with open(test_file, 'r', encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            wordtags = line.split()
            words = []
            gold_tags = []
            for wordtag in wordtags:
                word, tag = split_wordtag(wordtag)
                words.append(word)
                gold_tags.append(tag)
            sentence = ' '.join(words)
            predicted_tags = viterbi(sentence, transition, emission, possible_tags)
            predicted_tags = predicted_tags.split(' ')
            for t1, t2 in zip(predicted_tags, gold_tags):
                total += 1
                if t1 == t2:
                    correct += 1
    return 100.0 * correct/total, correct, total

acc, correct, total = evaluate('test.txt', transition, emission, possible_tags)
print("Accuracy = {} ({}/{})".format(acc, correct, total))
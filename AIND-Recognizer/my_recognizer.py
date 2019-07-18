import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # implement the recognizer
    for idx in range(test_set.num_items):
        best_prob, best_word = float("-inf"), None
        word_probs = {}
        seq, lengths = test_set.get_item_Xlengths(idx)
        for word, model in models.items():
            try:
                word_probs[word] = model.score(seq, lengths)
            except:
                word_probs[word] = float("-inf")

            if word_probs[word] > best_prob:
                best_prob, best_word = word_probs[word], word

        probabilities.append(word_probs)
        guesses.append(best_word)
    # return probabilities, guesses
    return probabilities, guesses

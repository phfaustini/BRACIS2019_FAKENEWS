#!/usr/bin/python
# -*- coding: utf-8 -*-

#### Class to perform a lexicon-based sentiment classification

#### Author: Pedro Paulo Balage Filho
#### Version: 1.0
#### Date: 05/12/12

#### Author: Pedro Henrique Arruda Faustini
#### Version: 1.1
#### Date: 13/11/18
#### Adapted to Python3

# performs a lexicon-based sentiment classification. You should initialize it
# with one of my dictionary classes: liwc, sentilex or opinionlexicon. They may
# have the polarity(word) method necessary by this class
class Classifier(object):


    # Constructor. Necessary to load a dictionary with the method
    # polarity(word) which returns 0,1,-1 or None
    def __init__(self, _dictionary):
        self.dictionary = _dictionary
        self.negators = ['não','nao','nunca','jamais','nada','tampouco','nenhum','nenhuma']
        self.modals = ['deve','pode','poderia','seria','deveria','seria']
        self.intensifiers = ['muito','demais','completamente','absolutamente','totalmente','definitivamente']
        self.intensifier_factor = 4

    # Measure the SO for the full sentence
    def classify(self,sentence):
        so_total,log = self.so_cal(sentence)
        if so_total > 0:
            return 1
        elif so_total < 0:
            return -1
        else:
            return 0

    # Measure the average polarity in the sentence
    def so_cal(self,sentence):

        so_total = 0.0
        log = ''
        negation,modal = False,False
        is_intensifier = False
        # window of next words in which the negation, modality or intensifier
        # operates
        neg_i = -10
        mod_i = -10
        int_i = -10

        # for each word in the sentence
        for i, w in enumerate(sentence):

            tag  = ''
            log += w
            w = w.lower()
            # Get the semantic orientation
            so = self.dictionary.polarity(w)

            if so:

                # previous Intensifiers
                if is_intensifier and (i-int_i)<=3:
                    so = so * self.intensifier_factor
                    log += '#Intensified'

                # previous negation
                if negation and (i-neg_i)<=3:
                    so = -so
                    log += '#Negated'

                # previous modal
                if modal and (i-mod_i)<=3:
                    so = 0
                    log += '#Irrealis'

                # Accumulate
                log += '#' + str(so)
                so_total += so

                # Reset variables
                negation = False
                modal = False
                is_intensifier = False
                intensifier = 0

            # word is a modal
            if w in self.modals:
                modal = True
                mod_i = i
                log +='#MODAL'
            # word is a negator
            if w in self.negators:
                negation = True
                neg_i = i
                log +='#NEGATION'
            # word is a intensifier
            if w in self.intensifiers:
                is_intensifier = True
                int_i = i
                log +='#INTENSIF'
            log += ' '
        log += '\n'

        return so_total,log

    # Analize the results
    def show_results(self,gold,test):
        from nltk import ConfusionMatrix
        correct = 0
        for index,result in enumerate(gold):
            if result == test[index]:
                correct +=1
        print ('Accuracy: {:.2%}'.format(float(correct) / float(len(gold))))
        cm = ConfusionMatrix(gold, test)
        print (cm.pp())


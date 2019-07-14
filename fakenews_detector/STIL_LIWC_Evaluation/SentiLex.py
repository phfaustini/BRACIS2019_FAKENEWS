# -*- coding: utf-8 -*-

#### Class to provide data and methods to read SentiLex dictionary
####
#### Author: Pedro Paulo Balage Filho
#### Version: 1.0
#### Date: 05/12/12

#### Author: Pedro Henrique Arruda Faustini
#### Version: 1.1
#### Date: 13/11/18
#### Adapted to Python3


import codecs
import re


class SentiLexReader(dict):

    """
    Dictionary format:

    à-vontade,à-vontade.PoS=N;FLEX=ms;TG=HUM:N0;POL:N0=1;ANOT=MAN
    abafada,abafado.PoS=Adj;FLEX=fs;TG=HUM:N0;POL:N0=-1;ANOT=JALC
    abafadas,abafado.PoS=Adj;FLEX=fp;TG=HUM:N0;POL:N0=-1;ANOT=JALC
    abafado,abafado.PoS=Adj;FLEX=ms;TG=HUM:N0;POL:N0=-1;ANOT=JALC
    abafados,abafado.PoS=Adj;FLEX=mp;TG=HUM:N0;POL:N0=-1;ANOT=JALC
    """

    # Constructor
    # dict_file: the path to dictionary file
    def __init__(self, dict_file='Dictionaries/SentiLex/SentiLex-flex-PT02.txt'):
        handle = codecs.open(dict_file, 'r', 'utf-8')

        line = handle.readline()
        prog = re.compile(r'([^,]*),([^.]*)\.PoS=([^;]*);([^;]*);TG=([^;]*);(POL:.*);ANOT=(.*)',re.I)
        while line:

        # Retrieve only the word/phrase and PoS
            m = prog.match(line)
            if m:
                phrase = m.group(1)
                lemma = m.group(2)
                pos = m.group(3)
                flex = m.group(4)
                target = m.group(5)
                polarities = m.group(6)
                anot = m.group(7)
                polarities = re.findall('POL:(N[0-9])=(-?[0-9])',polarities)
                polarities = [(srl,int(value)) for srl,value in polarities]
                if phrase in self:
                    self[phrase].append((pos,polarities))
                else:
                    self[phrase] = [(pos,polarities)]
            else:
                print (line)


            line = handle.readline()

        handle.close()

    #return all matches for a sentence which consists in list of words
    def match_words(self, sentence):

        i = 0
        length = len(sentence)
        j = length
        matches = []

        # iterate over the words present in the sentence
        while i < len(sentence):
            # get a slide window
            phrase = ' '.join(sentence[i:j])
            if i == j:
                i +=1
                j = length
            elif phrase in self:
                pos,polarities = self[phrase]
                matches.append( (phrase,pos,polarities) )
                i = j
                j = length
            else:
                j = j - 1

        return matches

    def print_statistics(self):
        return None

    def vocabulary(self):
        return set(self.keys())

    def vocabulary_polar(self):
        vocabulary = set()
        for key in self:
            if self.polarity(key) != 0:
                vocabulary.add(key)
        return vocabulary

    def polarity(self,word):
        if word in self:
            # how to select the polarity most representative among different
            # PoS and SRL?
            # I took the first PoS occurency and the fist SRL always
            return self[word][0][1][0][1]
        else:
            return None

    def get_name(self):
        return 'SentiLex'

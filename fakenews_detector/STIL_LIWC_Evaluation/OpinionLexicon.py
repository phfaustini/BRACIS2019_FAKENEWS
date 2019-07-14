# -*- coding: utf-8 -*-

#### Class to provide data and methods to read OpinionLexicon dictionary
#### Corpus reference:
####          Souza, M., Vieira, R., Chishman, R., & Alves, I. M. (2011).
####          Construction of a Portuguese Opinion Lexicon from multiple resources.
####          8th Brazilian Symposium in Information and Human Language Technology - STIL. Mato Grosso, Brazil.
####

#### Author: Pedro Paulo Balage Filho
#### Version: 1.0
#### Date: 05/12/12

#### Author: Pedro Henrique Arruda Faustini
#### Version: 1.1
#### Date: 13/11/18
#### Adapted to Python3

import codecs


class OpLexiconReader(dict):

    """
    Dictionary format:

    abalada,adj,-1
    abaladas,adj,-1
    abalado,adj,-1
    abalados,adj,-1
    abalançado,adj,1
    abalançar,vb,1
    abalar-se,vb,0
    abalar,vb,0
    abalizada,adj,1
    abalizadas,adj,1
    abalizado,adj,1
    abalizados,adj,1
    abalizar,vb,0
    abalroar-se,vb,1
    abalroar,vb,0

    """


    # Constructor
    # dict_file: the path to dictionary file
    def __init__(self, dict_file='Dictionaries/oplexicon/lexico_v2.1txt'):

        handle = codecs.open(dict_file, 'r', 'utf-8')

        # Run until the end of the file
        line = handle.readline()
        line = line.strip()
        while line:
            if not line.startswith('#') and not len(line)==0:

                phrase,pos,pol = line.split(',')
                try:
                    pol = int(pol)
                except:
                    print ('Error parsing the line:\n',line,'\n\n')

                if phrase in self:
                    self[phrase].append((pos,pol))
                else:
                    self[phrase] = [(pos,pol)]

            line = handle.readline()
            line = line.strip()

        handle.close()

    def get_name(self):
        return 'Opinion Lexicon'


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
            # PoS
            # I took the first PoS occurency
            return self[word][0][1]
        else:
            return None

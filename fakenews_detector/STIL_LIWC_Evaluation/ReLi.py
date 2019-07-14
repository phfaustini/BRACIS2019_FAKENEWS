# -*- coding: utf-8 -*-

#### Class to provide data and methods to read ReLi corpus
#### Corpus reference:
####          Freitas, C., Motta, E., Milidiú, R., & Cesar, J. (2012).
####          Vampiro que brilha... rá! Desafios na anotação de opinião em um corpus
####          de resenhas de livros. Proceedings do XI Encontro de Linguística de Corpus (XI ELC). São Carlos - SP.
####

#### Author: Pedro Paulo Balage Filho
#### Version: 1.0
#### Date: 05/12/12

#### Author: Pedro Henrique Arruda Faustini
#### Version: 1.1
#### Date: 13/11/18
#### Adapted to Python3

import re
import codecs
import os
from operator import itemgetter


# Class to provide data and methods to read ReLi corpus
class ReLiCorpusReader(dict):


    # Constructor
    # returns a dictionary containing the corpus
    # parameters:
    # path: the path to que corpus folder
    def __init__(self, path='Corpora/ReLi/'):
        corpus = self.build_corpus(path, 'all')
        for title in corpus:
            self[title] = corpus[title]

    # returns a dictionary containing the corpus
    # parameters:
    # path: the path to que corpus folder
    # output_format: format for the shallow level (values must be: word, pos or all)
    def build_corpus(self,path='Corpora/ReLi/',output_format = 'word'):

        corpus = {}

        # Check for parameter
        if output_format not in ['word','pos','all']:
            raise ValueError("output_format parameter must be: word, pos or all")

        #Append path slash if missing
        if path[-1]!='/':
            path += '/'

        # List all the files under the directory
        corpus_files = os.listdir(path)

        # Read each file and process
        for filename in corpus_files:
            # Filname pattern
            if filename.startswith('ReLi') and filename.endswith('.txt'):
                handle = codecs.open(path+filename,'r','utf-8')
                text = handle.readlines()
                handle.close()
                books = self.__reLiCorpusReader__(text,output_format)
                for title in books:
                    corpus[title] = books[title]

        return corpus

    # Function to read the column format from ReLi corpus to a dictionary data structure
    def __reLiCorpusReader__(self,text,output_format):
        # Corpus Structure
        """
        [features = word, pos, object, opinion, polarity, help]
        #Livro_Ensaio-Sobre-a-Cegueira
        #Resenha_0
        #Nota_3.0
        #Título_0
        Surreal	NPROP	O	O	O	O

        #Corpo_0
        Saramago	NPROP	O	O	O	O
        explora	V	O	O	O	O
        a	ART	O	O	O	O
        fantasia	N	O	O	O	O
        atrelada	PCP	O	O	O	O
        a	PREP	O	O	O	O
        cegueira	N	O	O	O	O
        de	PREP	O	O	O	O
        uma	ART	O	O	O	O
        maneira	N	O	O	O	O
        muito	ADV	O	O	O	O
        doida	ADJ	O	O	O	O
        .	.	O	O	O	O

        Ela	PROPESS	O	O	O	O
        alastra	V	O	O	O	O
        -	-	O	O	O	O
        se	PROPESS	O	O	O	O
        de	PREP	O	O	O	O
        maneira	N	O	O	O	O
        exponencial	ADJ	O	O	O	O
        .	.	O	O	O	O
        """

        # DataStructure to be used
        """
        --- books
        ------ book_title
        ------ reviews
        --------- review_id
        --------- score
        --------- review_title
        ------------ (word, pos, object, opinion, polarity, help)
        --------- sentences
        ------------ sent_words
        --------------- (word, pos, object, opinion, polarity, help)
        """

        # variables initialization
        books = {}
        reviews = {}
        review = {}
        sentences = []
        words = []
        review_id = -1
        book_title = 'NaN'

        # loop to go through the text
        for line in text:

            # Match the line specifying the book
            m = re.match(r"#Livro_(.+)",line)
            if m:
                # If words are not empty, append the last review read to the corpus
                if len(words) != 0:
                    sentences.append(words)
                    words = []

                # If sentences are not empty, append the last review read to the corpus
                if len(sentences) != 0:
                    reviews[review_id]['sentences'] = sentences
                    sentences = []

                # If reviews are not empty, append the last book read to the corpus
                if book_title != m.group(1).strip() and len(reviews) != 0:
                    books[book_title] = reviews
                    reviews = {}

                # book title
                book_title = m.group(1)
                book_title = book_title.strip()


            # Match the review_id
            m = re.match(r"#Resenha_([0-9]+)",line)
            if m:
                # If sentences are not empty, append the last review read to the corpus
                if len(sentences) != 0:
                    reviews[review_id]['sentences'] = sentences
                    sentences = []

                review_id = int(m.group(1))
                # Dictionary structure to keep this specific review internal attributes
                reviews[review_id] = {}


            # Match the score
            m = re.match(r"#Nota_([0-9.]+)",line)
            if m:
                reviews[review_id]['score'] = float(m.group(1))

            # Match the title, i.e., the sentences before the body (corpo)
            m = re.match(r"#Corpo_.+",line)
            if m:
                if len(sentences) != 0:
                    reviews[review_id]['title'] = sentences
                    sentences = []


            # Find the elements in each line (word, pos, object, opinion, polarity, help).
            m = re.match(r"([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t\[]+)[\n\[]",line)
            if m:
                word = unicode(m.group(1))
                pos = m.group(2)
                obj = m.group(3)
                opinion = m.group(4)
                pol = m.group(5)
                _help = m.group(6)

                # store as a tuple of items
                if 'word' in output_format:
                    words.append(word)
                if 'pos' in output_format:
                    words.append( (word,pos) )
                if 'all' in output_format:
                    words.append( (word,pos,obj,opinion,pol,_help) )

            # Match the break line. Sentence boundary
            if len(line.strip()) == 0:
                # Append the words read into the sentence
                if len(words) != 0:
                    sentences.append(words)
                    words = []

        # End (for line in text:)
        # End of file. Append the last words, sentence and reviews
        if len(words) != 0:
            sentences.append(words)
        reviews[review_id]['sentences'] = sentences
        books[book_title] = reviews

        # return the dictionary data containing the corpus data
        return books

    # Return corpus sentences. It is possible to filter by sentence polarity
    # The polarity assumes 3 values: neutral, positive, negative, polar, all. Any other value raises an error
    # neutral: return sentences without polarity
    # positive: positive sentences
    # negative: negative sentences
    # polar: positive or negative sentences
    # all: all sentences independent of their polarity
    # returns a list of sentences, where each sentence is a list of words. The words are tuples (word, pos, object, opinion, polarity, help).
    def sents(self, polarity='all'):

        # Check for parameter
        if polarity not in ['all','neutral','positive','negative','polar']:
            raise ValueError("polarity parameter must be: neutral, positive, negative, polar or all")

        # Variables initialization
        corpus = self
        sentences = []

        # iterate over the corpus
        for book in corpus:
            for review_id in corpus[book]:

                # check for sentences within the title. Those account in the same way as body sentences
                if 'title' in corpus[book][review_id]:
                    for sent in corpus[book][review_id]['title']:
                        # This magic analyses the 5th column (polarity) for the first tuple word in the sentence.
                        # In the corpus, the 5th column keeps the sentence polarity. It is the same for every word in the sentence
                        if sent[0][4] == '+' and polarity in ['positive','polar','all']:
                            sentences.append(sent)
                        if sent[0][4] == '-' and polarity in ['negative','polar','all']:
                            sentences.append(sent)
                        if sent[0][4] == 'O' and polarity in ['neutral','all']:
                            sentences.append(sent)


                # check for body sentences
                if 'sentences' in corpus[book][review_id]:
                    for sent in corpus[book][review_id]['sentences']:
                        # This magic analyses the 5th column (polarity) for the first tuple word in the sentence.
                        # In the corpus, the 5th column keeps the sentence polarity. It is the same for every word in the sentence
                        if sent[0][4] == '+' and polarity in ['positive','polar','all']:
                            sentences.append(sent)
                        if sent[0][4] == '-' and polarity in ['negative','polar','all']:
                            sentences.append(sent)
                        if sent[0][4] == 'O' and polarity in ['neutral','all']:
                            sentences.append(sent)

        # return the sentences which matched the polarity required
        return sentences

    # returns a list of words within a sentence structure.
    # In a sentence structure, each word is a tuple in the form (word, pos, object, opinion, polarity, help).
    def words_sentence(self,sentence):
        word_sentence = []
        for (word, pos, obj, opinion, pol, _help) in sentence:
            word_sentence.append(word)
        return word_sentence

    # returns a list of words within a sentence structure.
    # In a sentence structure, each word is a tuple in the form (word, pos, object, opinion, polarity, help).
    def words_sentence(self,sentence):
        word_sentence = []
        for (word, pos, obj, opinion, pol, _help) in sentence:
            word_sentence.append(word)
        return word_sentence

    # returns a list of words and PoS within a sentence structure.
    # In a sentence structure, each word is a tuple in the form (word, pos, object, opinion, polarity, help).
    def words_sentence_pos(self,sentence):
        word_sentence = []
        for (word, pos, obj, opinion, pol, _help) in sentence:
            word_sentence.append((word,pos))
        return word_sentence

    # pretty print the sentences structure printing word by word separating the words by a single space
    def pretty_print_sentences(self,sentences):
        for sent in sentences:
             print (' '.join(self.words_sentence(sent)))

    # returns the opinion aspects and predicates in the corpus
    # It is possible to filter by predicates according their polarity (parameters: positive, negative or polar)
    # returns a list of tuples in the form (aspect,predicate,polarity). Aspect and predicate are list of tuples in the format (word,PoS)
    def opinion_aspects(self,polarity='polar'):

        # check for the parameter
        if polarity not in ['positive','negative','polar']:
            raise ValueError("polarity parameter must be: positive, negative or polar")

        opinion_aspects = []
        corpus = self

        # iterate over the corpus
        for book in corpus:
            for review_id in corpus[book]:

                # these dictioanries will keep the information for this review_id
                aspects = {}
                predicates = {}
                polarities = {}

                # check for body sentences
                if 'sentences' in corpus[book][review_id]:
                    for sent in corpus[book][review_id]['sentences']:
                        for (word,pos,obj,opinion,pol,_help) in sent:

                            # Look for aspects
                            if opinion != 'O' or obj != 'O':
                                m1 = re.match(r"OBJ?([0-9]+)",obj)
                                m2 = re.match(r"OBJ?([0-9]+)",opinion)
                                m = 0
                                if m1:
                                    m = m1
                                if m2:
                                    m = m2
                                if m:
                                    aspect_id = int(m.group(1))
                                    if aspect_id not in aspects:
                                        aspects[aspect_id] = []
                                    # append the word to the aspects dictionary
                                    aspects[aspect_id].append((word,pos))

                            # Look for predicates
                            if opinion != 'O' or obj != 'O':
                                m1 = re.match(r"op([0-9]+)([+-])",opinion)
                                m2 = re.match(r"op([0-9]+)([+-])",obj)
                                m = 0
                                if m1:
                                    m = m1
                                if m2:
                                    m = m2
                                if m:
                                    signal = m.group(2)
                                    if (polarity == 'positive' and signal == '+') or (polarity == 'negative' and signal == '-') or (polarity == 'polar'):
                                        aspect_id = int(m.group(1))
                                        if aspect_id not in predicates:
                                            predicates[aspect_id] = []
                                        # append the word to the predicate dicationary
                                        predicates[aspect_id].append((word,pos))
                                        # append the polarity signal to the polarity dictionary
                                        polarities[aspect_id] = signal


                # check for sentences within the title. Those account in the same way as body sentences
                if 'title' in corpus[book][review_id]:
                    for sent in corpus[book][review_id]['title']:
                        for (word,pos,obj,opinion,pol,_help) in sent:

                            # Look for aspects
                            if opinion != 'O' or obj != 'O':
                                m1 = re.match(r"OBJ?([0-9]+)",obj)
                                m2 = re.match(r"OBJ?([0-9]+)",opinion)
                                m = 0
                                if m1:
                                    m = m1
                                if m2:
                                    m = m2
                                if m:
                                    aspect_id = int(m.group(1))
                                    if aspect_id not in aspects:
                                        aspects[aspect_id] = []
                                    # append the word to the aspects dictionary
                                    aspects[aspect_id].append((word,pos))

                            # Look for predicates
                            if opinion != 'O' or obj != 'O':
                                m1 = re.match(r"op([0-9]+)([+-])",opinion)
                                m2 = re.match(r"op([0-9]+)([+-])",obj)
                                m = 0
                                if m1:
                                    m = m1
                                if m2:
                                    m = m2
                                if m:
                                    signal = m.group(2)
                                    if (polarity == 'positive' and signal == '+') or (polarity == 'negative' and signal == '-') or (polarity == 'polar'):
                                        aspect_id = int(m.group(1))
                                        if aspect_id not in predicates:
                                            predicates[aspect_id] = []
                                        # append the word to the predicate dicationary
                                        predicates[aspect_id].append((word,pos))
                                        # append the polarity signal to the polarity dictionary
                                        polarities[aspect_id] = signal


                # aspect 0, when not present, is reserved to the book entity
                if 0 not in aspects:
                    aspects[0] = [('#book#','N')]

                # Compile all values found in the opinion_aspects list
                for aspect_id in predicates:
                    opinion_aspects.append((aspects[aspect_id],predicates[aspect_id],polarities[aspect_id]))

        return opinion_aspects


    # pretty print the aspects list
    def pretty_print_aspects(self,aspects):
        for (aspect,predicate,signal) in aspects:
            print (signal, ' '.join([w for w,p in aspect]),':',(40-len(' '.join([w for w,p in aspect])))*' ', ' '.join([w for w,p in predicate]))


    # Creates a frequency list
    # Returns a list of tuples in the form (item,frequency) sorted by the frequency of their the items in a list
    def __freq_list__(self,items):
        freq = {}
        for item in items:
            if isinstance(item,str):
                item = item.lower()
            freq[item] = freq.get(item,0) + 1

        return [(item,n) for item, n in sorted(freq.items(), key=itemgetter(1), reverse=True)]

    # Prints some statistics about the corpus
    def print_statistics(self):

        # parameters for pretty printing
        spaces = 25
        max_items = 20

        freq_list = self.__freq_list__

        # Compute lists for words, sentences, predicates and aspects

        # Compute all sentences and words present in the corpus
        sentences = self.sents('all')
        num_words = 0
        all_words = []
        for sent in sentences:
            all_words += self.words_sentence(sent)


        # Compute all aspects and predicates
        all_aspects = self.opinion_aspects('polar')

        # initialize variables
        predicates = {}
        aspects = {}
        predicates['pos'] = []
        predicates['neg'] = []
        predicates['all'] = []
        aspects['pos'] = []
        aspects['neg'] = []
        aspects['all'] = []

        # append the aspects and predicates according the polarity signal
        for (aspect,predicate,polarity) in all_aspects:

            if polarity == '+':
                 predicates['pos'].append(' '.join([w for w,p in predicate]))
                 aspects['pos'].append(' '.join([w for w,p in aspect]))

            if polarity == '-':
                 predicates['neg'].append(' '.join([w for w,p in predicate]))
                 aspects['neg'].append(' '.join([w for w,p in aspect]))

            predicates['all'].extend(predicate)
            aspects['all'].extend(aspect)


        # print statistics
        print ('::::::: General Statistics ::::::::\n')

        # Number of words:
        print( 'Number of words: {0}'.format( len(all_words) ) )

        # Number of sentences:
        print( 'Number of sentences: {0}'.format( len(sentences) ) )

        # Words most frequent in the corpus
        print( 'Most frequent words in the corpus:')
        for item,n in freq_list(all_words)[:max_items]:
            print (item, (spaces-len(item))*' ', n)


        print ('\n\n::::::: Sentence Statistics ::::::::\n')

        # Sentence Polarity
            # Number of positive sentences
        print( 'Number of positive sentences: {0}'.format( len(self.sents('positive') ) ) )
            # Number of negative sentences
        print( 'Number of negative sentences: {0}'.format( len(self.sents('negative') ) ) )
            # Number of neutral sentences
        print( 'Number of neutral sentences:  {0}'.format( len(self.sents('neutral') ) ) )

        print ('\n\n::::::: Aspects/Object Statistics ::::::::\n')

        # Most frequent aspects words in the corpus and their PoS
        print( 'Most frequent words present in aspects and their PoS:')
        for (word,pos),n in freq_list(aspects['all'])[:max_items]:
            print (word + '/' + pos, (spaces-len(word+pos))*' ', n)

        # Number of positive and negative predicates.
        print( '\n\nNumber of positive aspects: {0}'.format( len(aspects['pos'] ) ) )
        print( 'Number of negative aspects: {0}'.format( len(aspects['neg'] ) ) )

        # Most frequent aspects in the corpus
        print( '\nMost frequent positive aspects in the corpus:')
        for item,n in freq_list(aspects['pos'])[:max_items]:
            print (item, (spaces-len(item))*' ', n)

        print( '\n\nMost frequent negative aspects in the corpus:')
        for item,n in freq_list(aspects['neg'])[:max_items]:
            print (item, (spaces-len(item))*' ', n)


        print ('\n\n::::::: Predicates/Opinions Statistics ::::::::\n')

        # Most frequent predicate words in the corpus and their PoS
        print( 'Most frequent words present in predicates and their PoS:')
        for (word,pos),n in freq_list(predicates['all'])[:max_items]:
            print (word + '/' + pos, (spaces-len(word+pos))*' ', n)


        print( '\n\nNumber of positive predicates: {0}'.format( len(predicates['pos'] ) ) )
        print( 'Number of negative predicates: {0}'.format( len(predicates['neg'] ) ) )

        # Most frequent predicates in the corpus
        print( '\nMost frequent positive predicates in the corpus:')
        for item,n in freq_list(predicates['pos'])[:max_items]:
            print (item, (spaces-len(item))*' ', n)

        print( '\n\nMost frequent negative predicates in the corpus:')
        for item,n in freq_list(predicates['neg'])[:max_items]:
            print (item, (spaces-len(item))*' ', n)


    # Function to convert the lines present in the txt file into a html representation
    def __text2html__(self,text):
        html = '''
        <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
        <html xmlns="http://www.w3.org/1999/xhtml">
        <head>
            <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
        </head>
        '''
        livro = 'NaN'
        for line in text:

            # Match the line specifying the book
            m = re.match(r"#Livro_(.+)$",line)
            if m:
                if m.group(1) != livro:
                    html = html + '<h3>Livro:' + m.group(1) + '</h3>'
                else:
                    livro = m.group(1)
                livro = m.group(1)

            # Match the review
            m = re.match(r"#Resenha_([0-9]+)",line)
            if m:
                html = html + '<br/><b>Resenha:' + m.group(1) + '</b><br/>'

            # Match the score
            m = re.match(r"#Nota_([0-9.]+)",line)
            if m:
                html = html + '<b>Nota:' + m.group(1) + '</b><br/>'

            # Match the score
            m = re.match(r"#Título_(.+)$",line)
            if m:
                html = html + '<b>Título:' + m.group(1) + '</b><br/>'

            # An empty line is a break line
            if len(line.strip()) == 0:
                html = html + '<br/>'

            # Find the elements in each line (word, pos, object, opinion, polarity, help)
            m = re.match(r"([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t\[]+)[\n\[]",line)
            if m:
                word = unicode(m.group(1))
                pos = m.group(2)
                obj = m.group(3)
                opinion = m.group(4)
                pol = m.group(5)
                _help = m.group(6)

                if obj != 'O':
                    html = html + '<b><a title="'+obj+'">' + word + '</a></b> '
                elif opinion != 'O':
                    if opinion.endswith('-'):
                        html = html + '<font color="red"><a title="' + opinion + '">' + word + '</a></font> '
                    else:
                        html = html + '<font color="blue"><a title="' + opinion + '">' + word + '</a></font> '
                else:
                    html = html + word + ' '
        html = html + '</html>'
        return html


    def convertHtml(self, path='Corpora/ReLi/',output_path='Corpora/ReLiWeb/'):

        #Append path slash
        if path[-1]!='/':
            path += '/'

        if output_path[-1]!='/':
            Output_path += '/'

        corpus_files = os.listdir(path)

        if not os.path.exists(output_path):
            os.mkdir(output_path)

        for filename in corpus_files:
            # Filname pattern
            if filename.startswith('ReLi') and filename.endswith('.txt'):
                handle = codecs.open(path+filename,'r','utf-8')
                text = handle.readlines()
                handle.close()

                html = self.__text2html__(text)

                handle = codecs.open(output_path+filename[:-3]+'html','w','utf-8')
                handle.write(html)

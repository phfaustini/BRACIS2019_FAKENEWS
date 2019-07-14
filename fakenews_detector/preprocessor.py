import os
from glob import glob
from fileinput import input

import nltk
from nltk.tokenize import RegexpTokenizer
import numpy as np
import pandas as pd
import hunspell

from .STIL_LIWC_Evaluation.Liwc import LiwcReader
from .STIL_LIWC_Evaluation.LexiconClassifier import Classifier as PortuguesePolarityClassifier

from .news import Text


class PreProcessor():

    """
    Convert raw to structured data.
    """

    def __init__(self, platform_folder='datasets/WhatsApp/whats_br/'):
        self.PT_BR_dic = '/usr/share/hunspell/pt_BR.dic'
        self.PT_BR_aff = '/usr/share/hunspell/pt_BR.aff'
        self.spell_checker = hunspell.HunSpell(self.PT_BR_dic, self.PT_BR_aff)
        self.PLATFORM_FOLDER = platform_folder
        self.NILC_CORPUS = "Corpus/corpus100.txt"
        self.tagged_sentences = []  # [ [('word', 'type'), ('word', 'type'), ...], [('word', 'type'), ...], ...   ]
        # Train a POS tag model from nilc corpus.
        for line in input(self.NILC_CORPUS):
            tsent = []
            line_l = line.split(" ")
            for w in line_l:
                w = w.split("_")
                if len(w) > 1:
                    tsent.append((w[0], w[1]))
            self.tagged_sentences.append(tsent)
        t0 = nltk.DefaultTagger('NN')
        t1 = nltk.UnigramTagger(self.tagged_sentences, backoff=t0)
        t2 = nltk.BigramTagger(self.tagged_sentences, backoff=t1)
        self.tagger = nltk.TrigramTagger(self.tagged_sentences, backoff=t2)
        self.polarity_classifier = PortuguesePolarityClassifier(LiwcReader())

    def _text_2_list_of_list_of_strings(self, text: str) -> list:
        """
        Break sentences into a list of lists of
        strings.

        Example:

            _text_2_list_of_list_of_strings('First sentence. Now the second one.')

            Returns: [['First','sentence'], ['Now','the','second','one']]

        :param text: any string.
        :return: a list  of lists. Each sublist
        contains words (str format).
        """
        sentences = nltk.sent_tokenize(text, language='portuguese')
        strings = []
        tokenizer = RegexpTokenizer(r'\w+')
        for s in sentences:
            strings.append(tokenizer.tokenize(s))
        return strings

    def get_words_per_sentence(self, text: str) -> float:
        """Given a text, returns how many words, in average,
        sentences have.

        :param text: a string.
        :return: a float, with the average size of the sentences.
        """
        return np.mean(list(map(len, self._text_2_list_of_list_of_strings(text))))

    def _polarity_sentence(self, sentence: list) -> int:
        """
        This method tells the polarity of a given sentence.
        The type of sentence depends on the language, because
        each language uses a different model to compute polarity.

        Polarity is one of the following:
            -1: Negative
             0: Neutral
            +1: Positive

        example: sentence = ['Muito', 'bom', 'texto']
                 return: +1

        :param sentence: a list of strings, or tokens (portuguese)
                         a string (english)
        :return: 0, 0.5 or 1
        """
        polarity = self.polarity_classifier.classify(sentence)
        if polarity == -1:
            return 0
        elif polarity == 0:
            return 0.5
        else:
            return 1

    def polarity_text(self, text: str) -> float:
        """
        This method tells the polarity of a given text.
        Polarity is one of the following:
            0.0: Negative
            0.5: Neutral
            1.0: Positive

        example: text = 'Primeira frase. Agora, a segunda frase foi muito ruim.'
                 return: 0.25

        It measures the polarity of the text by simply taking the
        mean polarity of the sentences.

        :param text: a string
        :return: [0 .. 1].
        """
        return np.mean(list(map(self._polarity_sentence, self._text_2_list_of_list_of_strings(text))))

    def get_proportion_spell_error(self, text: str) -> float:
        """Returns the proportion of words wrongly spelled
        in the given text.

        :param text: a string.
        :return: [0..1].
        """
        tokenizer = RegexpTokenizer(r'\w+')
        words = tokenizer.tokenize(text)
        return len(list(filter(lambda word: self.spell_checker.spell(word) is False, words))) / len(words)

    def get_lexical_size(self, text: str) -> int:
        """Returns how many unique words there
        are in the given text.
        """
        tokenizer = RegexpTokenizer(r'\w+')
        words = tokenizer.tokenize(text)
        return len(set(words))

    def get_swear_words(self, text: str) -> int:
        """Returns how many times swear words appear
        in the given text. Swear words is a closed set
        of words defined here beforehand.
        """
        tokenizer = RegexpTokenizer(r'\w+')
        words = list(map(lambda word: word.lower(), tokenizer.tokenize(text)))
        swear_words = ['puto', 'fdp', 'porra', 'caralho', 'vigarista', 'hipócrita', 'merda', 'fudeu', 'fuder', 'foda', 'fuderam', 'cu', 'bosta']
        return len(list(filter(lambda word: word in swear_words, words)))

    def get_proportion_tag(self, text: str, tag='ADJ') -> float:
        """
        Returns the proportion of tokens in a given text
        that are of type <tag>.

        For example, if <tag> is VTI, and 3 of 10 tokens are VTI,
        it will return 0.3
        :param text: any string.
        :param tag: string, such as ADJ, ADV, N, VTD_PPR...
        """
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(text)
        word_token = self.tagger.tag(tokens)  # [ (word0, token0), (word1, token1), ... ]
        return len(list(filter(lambda x: x[1] == tag, word_token))) / len(word_token)

    def _load_file(self, filepath: str) -> dict:
        wp_dict = {'text': "", 'filepath': ""}
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                wp_dict['text'] = f.read()
                wp_dict['filepath'] = filepath
        except UnicodeDecodeError:
            with open(filepath, 'r', encoding='cp1252') as f:
                wp_dict['text'] = f.read()
                wp_dict['filepath'] = filepath
        return wp_dict

    def _convert_json_to_obj(self, json: dict) -> Text():
        obj = Text()
        text = json['text']
        uppercase_letters_count = len(list(filter(lambda x: str.isupper(x), text)))
        lowercase_letters_count = len(list(filter(lambda x: str.islower(x), text)))
        exclamation_marks_count = len(list(filter(lambda x: x == "!", text)))
        question_marks_count = len(list(filter(lambda x: x == "?", text)))
        total_letters = uppercase_letters_count + lowercase_letters_count
        obj.uppercase_letters = (uppercase_letters_count / total_letters)
        obj.lowercase_letters = (lowercase_letters_count / total_letters)
        obj.swear_words = self.get_swear_words(text)
        obj.exclamation_marks = (exclamation_marks_count / (total_letters + exclamation_marks_count))
        if obj.exclamation_marks > 0:
            obj.has_exclamation = 1.0
        else:
            obj.has_exclamation = 0.0
        obj.question_marks = (question_marks_count / (total_letters + question_marks_count))
        obj.words_per_sentence = self.get_words_per_sentence(text)
        obj.ADJ = self.get_proportion_tag(text, 'ADJ')
        obj.ADV = self.get_proportion_tag(text, 'ADV')
        obj.N = self.get_proportion_tag(text, 'N')
        obj.spell_errors = self.get_proportion_spell_error(text)
        obj.lexical_size = self.get_lexical_size(text)
        obj.polarity = self.polarity_text(text)
        obj.text = text
        obj.number_sentences = float(len(nltk.sent_tokenize(obj.text)))
        obj.len_text = float(len(text))
        return obj

    def convert_rawdataset_to_dataset(self, platform='WhatsApp', dataset='whats_br', class_label='False'):
        """
        This method sould parse all .txt content in Raw and store it Structured folders in .csv format.
        """
        counter = 1
        for filepath in sorted(glob('datasets/{0}/{1}/Raw/{2}/*/*'.format(platform, dataset, class_label))):
            obj_dict = self._load_file(filepath=filepath)
            obj = self._convert_json_to_obj(obj_dict)
            if obj is not None:
                df = self.convert_obj_to_dataframe(label=class_label, obj=obj)
                queryfolder = "{0}/{1}/".format(class_label, filepath.split("/")[5])
                self.write_to_disk(queryfolder, class_label, counter, df)
                counter += 1
            else:
                pass  # print("convert_rawdataset_to_dataset: failed to convert {0}".format(filepath))

    def convert_obj_to_dataframe(self, label: str, obj: Text) -> pd.DataFrame():
        return pd.DataFrame(
                            [
                              [
                                round(obj.uppercase_letters, 2),
                                round(obj.exclamation_marks, 2),
                                obj.has_exclamation,
                                round(obj.question_marks, 2),
                                round(obj.words_per_sentence, 2),
                                round(obj.ADJ, 2),
                                round(obj.ADV, 2),
                                round(obj.N, 2),
                                round(obj.spell_errors, 2),
                                round(obj.lexical_size, 2),
                                obj.polarity,
                                obj.number_sentences,
                                obj.len_text,
                                obj.swear_words,
                                # obj.text.replace('\n', ' '),
                                1.0 if label == "False" else -1.0
                              ]
                            ],
                            columns=["uppercase", "exclamation", "has_exclamation", "question", "words_per_sentence", "adj", "adv", "noun", "spell_errors", "lexical_size", "polarity", "number_sentences", "len_text", "swear_words", "label"]
                           )

    def write_to_disk(self, queryfolder: str, label: str, counter: int, df_obj: pd.DataFrame):
        """Save the object to disk, into folder
        datasets/<Twitter,Websites,WhatsApp>/<Something>/

        Files are saved in format <label>-<counter>.csv
        Values in each line are separated by comma (',').

        :param queryfolder: a string ending with '/' (e.g. 'False/Some fake news/')
        :param label: a string with the class of the object.
        :param counter: a unique counting identifier for naming purposes.
        :param df_obj: a pd.DataFrame with structured objects.
        """
        filename = "{0}-{1}.csv".format(label, counter)
        print("write_to_disk {0}".format(filename))
        if not os.path.isdir("{0}Structured/".format(self.PLATFORM_FOLDER)):
            os.mkdir("{0}Structured/".format(self.PLATFORM_FOLDER))
        if not os.path.isdir("{0}Structured/{1}".format(self.PLATFORM_FOLDER, label)):
            os.mkdir("{0}Structured/{1}".format(self.PLATFORM_FOLDER, label))
        if not os.path.isdir("{0}Structured/{1}".format(self.PLATFORM_FOLDER, queryfolder)):
            os.mkdir("{0}Structured/{1}".format(self.PLATFORM_FOLDER, queryfolder))
        df_obj.to_csv("{0}Structured/{1}{2}".format(self.PLATFORM_FOLDER, queryfolder, filename), index=False)

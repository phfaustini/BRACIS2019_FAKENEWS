import os
import sys
from ast import literal_eval
import json

import nltk
import numpy as np
import pandas as pd

from .preprocessor import PreProcessor
from .news import Tweet, User


class PreProcessorTwitter(PreProcessor):

    """
    Convert semi-structured (json) to class objects.
    """

    def __init__(self, platform_folder='datasets/Twitter/tweets_br/'):
        PreProcessor.__init__(self, platform_folder=platform_folder)

    def _load_file(self, filepath: str) -> dict:
        """
        Given the filepath of a file, this method
        returns the file's json representation as a dict.

        :param filepath: path of the file.
        Example of param: 'datasets/Twitter/tweets_br/Raw/False/Fake topic/1020479528047726593.txt'
        """
        print("_load_file: beginning of {0}".format(filepath))
        json_data = {}
        try:
            if os.path.isfile(filepath):
                with open(filepath, "r", encoding='utf-8') as f:
                    f_s = f.read()
                    json_data = json.loads(f_s)
            else:
                print("File {0} not found!".format(filepath))
        except ValueError:
            try:
                json_data = literal_eval(f_s)
            except ValueError:
                print("Error while reading Json {0}".format(filepath))
        except:
            print("Unknown error: "+str(sys.exc_info()[0]))
        return json_data

    def _convert_json_to_obj(self, json: dict) -> Tweet:
        try:
            obj = Tweet()
            uppercase_letters = len(list(filter(lambda x: str.isupper(x), json['full_text'])))
            lowercase_letters = len(list(filter(lambda x: str.islower(x), json['full_text'])))
            exclamation_marks = len(list(filter(lambda x: x == "!", json['full_text'])))
            question_marks = len(list(filter(lambda x: x == "?", json['full_text'])))
            total_letters = uppercase_letters + lowercase_letters
            obj.full_text.uppercase_letters = (uppercase_letters / total_letters)
            obj.full_text.lowercase_letters = (lowercase_letters / total_letters)
            obj.full_text.exclamation_marks = (exclamation_marks / (total_letters + exclamation_marks))
            if obj.full_text.exclamation_marks > 0:
                obj.full_text.has_exclamation = 1
            else:
                obj.full_text.has_exclamation = 0
            obj.full_text.question_marks = (question_marks / (total_letters + question_marks))
            obj.full_text.ADJ = self.get_proportion_tag(json['full_text'], 'ADJ')
            obj.full_text.ADV = self.get_proportion_tag(json['full_text'], 'ADV')
            obj.full_text.N = self.get_proportion_tag(json['full_text'], 'N')
            obj.id_str = json['id_str']
            obj.full_text.text = json['full_text']
            obj.full_text.spell_errors = self.get_proportion_spell_error(obj.full_text.text)
            obj.full_text.polarity = self.polarity_text(obj.full_text.text)
            obj.full_text.number_sentences = float(len(nltk.sent_tokenize(obj.full_text.text)))
            obj.full_text.len_text = float(len(json['full_text']))
            obj.full_text.words_per_sentence = self.get_words_per_sentence(obj.full_text.text)
            obj.full_text.lexical_size = self.get_lexical_size(obj.full_text.text)
            obj.full_text.swear_words = self.get_swear_words(obj.full_text.text)
            obj.contributors = json['contributors']
            obj.coordinates = json['coordinates']
            obj.created_at = json['created_at']
            obj.display_text_range = json['display_text_range']
            obj.entities = json['entities']
            obj.favorite_count = json['favorite_count']
            obj.favorited = json['favorited']
            obj.geo = json['geo']
            obj.is_quote_status = json['is_quote_status']
            obj.lang = json['lang']
            obj.place = json['place']
            obj.retweet_count = json['retweet_count']
            obj.retweeted = json['retweeted']
            obj.truncated = json['truncated']
            obj.urls = list(map(lambda urls_dict: urls_dict['expanded_url'], json['entities']['urls']))
            obj.number_urls = float(len(obj.urls))
            obj.user = User()
            obj.user.verified = json['user']['verified']
            obj.user.id_str = json['user']['id_str']
            obj.user.name = json['user']['name']
            obj.user.description = json['user']['description']
            obj.user.location = json['user']['location']
            obj.user.protected = json['user']['protected']
            obj.user.friends_count = json['user']['friends_count']
            obj.user.followers_count = json['user']['followers_count']
            obj.user.geo_enabled = json['user']['geo_enabled']
            obj.user.lang = json['user']['lang']
            obj.user.created_at = json['user']['created_at']
            return obj
        except:
            return None

    def convert_obj_to_dataframe(self, label: str, obj: Tweet) -> pd.DataFrame():
        """
        Convert an object to a pd.DataFrame

        :param label: a string with the class of the object.
        :param obj: a Tweet or Text object.
        :returns: a pd.DataFrame with a list of objects.
        """
        return pd.DataFrame(
                            [
                              [
                                round(obj.full_text.uppercase_letters, 2),
                                round(obj.full_text.exclamation_marks, 2),
                                round(obj.full_text.question_marks, 2),
                                obj.full_text.has_exclamation,
                                round(obj.full_text.words_per_sentence, 2),
                                round(obj.full_text.ADJ, 2),
                                round(obj.full_text.ADV, 2),
                                round(obj.full_text.N, 2),
                                round(obj.full_text.spell_errors, 2),
                                round(obj.full_text.lexical_size, 2),
                                obj.full_text.polarity,
                                obj.full_text.number_sentences,
                                obj.full_text.len_text,
                                obj.full_text.swear_words,
                                # obj.full_text.text.replace('\n', ' '),
                                1.0 if label == "False" else -1.0
                              ]
                            ],
                            columns=["uppercase", "exclamation", "has_exclamation", "question", "words_per_sentence", "adj", "adv", "noun", "spell_errors", "lexical_size", "polarity", "number_sentences", "len_text", "swear_words", "label"]
                           )

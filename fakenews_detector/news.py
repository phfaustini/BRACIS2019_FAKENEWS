class Text():

    """Class representation for textual features."""

    def __init__(self):
        self.lowercase_letters = 0.0
        self.uppercase_letters = 0.0
        self.exclamation_marks = 0.0
        self.question_marks = 0.0
        self.has_exclamation = 0.0
        self.lexical_size = 0.0
        self.ADV = 0.0
        self.ADJ = 0.0
        self.N = 0.0
        self.spell_errors = 0.0
        self.text = None
        self.polarity = 0
        self.number_sentences = 0
        self.len_text = 0
        self.words_per_sentence = 0.0
        self.swear_words = 0.0


class User():

    """Class representation for user features."""

    def __init__(self):
        self.id_str = None
        self.name = None
        self.screen_name = None
        self.location = None
        self.location = None
        self.description = None
        self.url = None
        self.entities = None
        self.protected = None
        self.followers_count = None
        self.friends_count = None
        self.listed_count = None
        self.created_at = None
        self.favourites_count = None
        self.utc_offset = None
        self.time_zone = None
        self.geo_enabled = None
        self.verified = None
        self.statuses_count = None
        self.lang = None
        self.contributors_enabled = None
        self.is_translator = None
        self.is_translation_enabled = None
        self.profile_background_color = None
        self.profile_background_image_url = None
        self.profile_background_image_url_https = None
        self.profile_background_tile = None
        self.profile_image_url = None
        self.profile_image_url_https = None
        self.profile_link_color = None
        self.profile_sidebar_border_color = None
        self.profile_sidebar_fill_color = None
        self.profile_text_color = None
        self.profile_use_background_image = None
        self.has_extended_profile = None
        self.default_profile = None
        self.default_profile_image = None
        self.following = None
        self.follow_request_sent = None
        self.notifications = None
        self.translator_type = None


class Tweet():

    """Class representation for tweet features."""

    def __init__(self):
        self.id_str = None
        self.full_text = Text()
        self.user = User()
        self.created_at = None
        self.truncated = None
        self.display_text_range = None
        self.entities = None
        self.geo = None
        self.coordinates = None
        self.place = None
        self.contributors = None
        self.is_quote_status = None
        self.retweet_count = None
        self.favorite_count = None
        self.favorited = None
        self.retweeted = None
        self.lang = None
        self.urls = []
        self.number_urls = None

from LexiconClassifier import Classifier
from Liwc import LiwcReader
from OpinionLexicon import OpLexiconReader
from SentiLex import SentiLexReader

l = LiwcReader()
s = SentiLexReader()
o = OpLexiconReader()
c = Classifier(o)

r = c.classify(["corrupto"])
print(r)
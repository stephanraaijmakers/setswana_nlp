import sys
import nltk
from nltk.collocations import *
from nltk import word_tokenize
from nltk.collocations import BigramCollocationFinder


bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()

with open(sys.argv[1],"r") as fp:
    words=[w for l in fp.readlines() for w in l.split(" ")]

finder = BigramCollocationFinder.from_words(words)
print(finder.nbest(bigram_measures.raw_freq, 100))


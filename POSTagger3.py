# Based on: https://github.com/neubig/lxmls-2017/blob/master/postagger.py
# POS Tagger that concatenates word embeddings with character-level embeddings to represent words, and generates tags

#                       --> lookup table --> we     --> biLSTM  --> we2
# [word1, word2, ...]                                                       --> [we + we2_c] --> MLP --> tags
#                       --> lookup table --> we_c   --> biLSTM  --> we2_c


# TODO implement this option?


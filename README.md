# dynet_toy_examples

## POSTagger1

Allows to use an embeddings lookup table or a character-based bi-LSTM:

    [word1, word2, ...] --> lookup table or char_bilstm --> we --> biLSTM  --> MLP --> tags

## POSTagger2

Concatenates word embeddings with character-level embeddings to represent words, and feeds them through a biLSTM to encode the words and generate tags.

NOTICE that we need the char biLSTM because otherwise we_c would be an embedding for each character, and we need something with fixed size!

NOTICE how each sequence has a different length, but its OK, the LstmAcceptor doesn’t care. We create a new graph  for each example, at exactly the desired length.

                          --> lookup table --> we
    [word1, word2, ...]                                                                     --> [we + we2_c] --> biLSTM --> MLP --> tags
                          --> char biLSTM [lookup table --> we_c --> biLSTM   --> we2_c]

## POSTagger3

Identical to POSTagger2, but it allows **minibatching**.

## POSTagger4

POSTagger4 concatenates word embeddings extracted from a graph-based parser with character-level embeddings to represent words, and feeds them through a biLSTM to encode the words and generate tags.  

More precisely, we parse the input sentence and for each word concatenate the internal states of the biLSTM

                          --> K&G parser    --> we
    [word1, word2, ...]                                                                         --> [we + we2_c] --> biLSTM --> MLP --> tags
                          --> char biLSTM [lookup table  --> we_c --> biLSTM   --> we2_c]


## TODO:
- improve model saving with best three model saving. Is there something automatic?
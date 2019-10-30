# dynet_toy_examples

# models

## POSTagger1

Allows to use an embeddings lookup table or a character-based bi-LSTM:

    [word1, word2, ...] --> lookup table or char_bilstm --> we --> biLSTM  --> MLP --> tags

- with lookup table:
    - Test accuracy: 0.8949276805992749
    - Elapsed time: 192.212525121
- with char LSTM:
    - Test accuracy: 0.8724150296848229
    - Elapsed time: 529.6719069080001

## POSTagger2

Concatenates word embeddings with character-level embeddings to represent words, and feeds them through a biLSTM to encode the words and generate tags.

NOTICE that we need the char biLSTM because otherwise we_c would be an embedding for each character, and we need something with fixed size!

NOTICE how each sequence has a different length, but its OK, the LstmAcceptor doesn’t care. We create a new graph  for each example, at exactly the desired length.

                          --> lookup table --> we
    [word1, word2, ...]                                                                     --> [we + we2_c] --> biLSTM --> MLP --> tags
                          --> char biLSTM [lookup table --> we_c --> biLSTM   --> we2_c]

- Test accuracy: 0.9168426505159979
- Elapsed time: 921.129285493

## POSTagger3

Identical to POSTagger2, but it allows **minibatching**.

- Test accuracy: 0.9081961987488545
- Elapsed time: 988.706079337

## POSTagger4

POSTagger4 concatenates word embeddings extracted from a graph-based parser with character-level embeddings to represent words, and feeds them through a biLSTM to encode the words and generate tags.  

More precisely, we parse the input sentence and for each word concatenate the internal states of the biLSTM

                          --> K&G parser    --> we
    [word1, word2, ...]                                                                         --> [we + we2_c] --> biLSTM --> MLP --> tags
                          --> char biLSTM [lookup table  --> we_c --> biLSTM   --> we2_c]

- K&G mini_test:
    - Test accuracy: 0.8416145356018647
    - Elapsed time: 2531.506128486
- K&G mini:
    - Test accuracy: 0.9109455313384069
    - Elapsed time: 2598.515521656

# TODO:
- improve model saving with best three model saving. Is there something automatic?
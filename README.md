# dynet_toy_examples

# models

## POSTagger1

Allows to use an embeddings lookup table or a character-based bi-LSTM:

    [word1, word2, ...] --> lookup table or char_bilstm --> we --> biLSTM  --> MLP --> tags

- with lookup table:
    - Test accuracy: 0.8949276805992749
    - Elapsed time: 192.212525121
- with char biLSTM:
    - Test accuracy: 0.8724150296848229
    - Elapsed time: 529.6719069080001

## POSTagger2

Concatenates word embeddings with character-level embeddings to represent words, and feeds them through a biLSTM to encode the words and generate tags.

NOTICE that we need the char biLSTM because otherwise we_c would be an embedding for each character, and we need something with fixed size!

NOTICE how each sequence has a different length, but its OK, the LstmAcceptor doesn’t care. We create a new graph  for each example, at exactly the desired length.

                          --> lookup table --> we
    [word1, word2, ...]                                                                     --> [we + we2_c] --> biLSTM --> MLP --> tags
                          --> char biLSTM [lookup table --> we_c --> biLSTM   --> we2_c]

- without autobatching:
    - Test accuracy: 0.9168426505159979
    - Elapsed time: 921.129285493
- with autobatching:
    - Test accuracy: 0.9113439853368929
    - Elapsed time: 646.87033149

## POSTagger3

Identical to POSTagger2, but it allows **minibatching**.

- batch 32:
    - Test accuracy: 0.9081961987488545
    - Elapsed time: 988.706079337
- batch 64:
    - Test accuracy: 0.8979160855879189
    - Elapsed time: 639.832461975
- batch 128:
    - Test accuracy: 0.8738096186795234
    - Elapsed time: 678.990824393

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


# Observations:

- a we lookup table is faster and more accurate than char biLSTM, but
    - a combination of both is better
- autobatching effectively reduces training time without reducing accuracy
- minibatching:
    - does not help much in comparison with just using autobatching
    - batches of 64 are faster than 32, but
        - batches of 128 are slower than 64
- K&G mini_test is too small (small vocabulary) to produce high accuracy


# TODO:

- improve model saving with best three model saving. Is there something automatic?
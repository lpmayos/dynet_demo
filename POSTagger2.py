import logging
import os
import time
import dynet as dy
from POSTagger import POSTagger, UNK_TOKEN


class POSTagger2(POSTagger):
    """
    POSTagger2 concatenates word embeddings with character-level embeddings to represent words, and feeds them through a
    biLSTM to encode the words and generate tags.

    NOTICE that we need the char biLSTM because otherwise we_c would be an embedding for each character, and we need
    something with fixed size!
    NOTICE how each sequence has a different length, but its OK, the LstmAcceptor doesn’t care. We create a new graph
    for each example, at exactly the desired length.

                          --> lookup table --> we
    [word1, word2, ...]                                                       --> [we + we2_c] --> biLSTM --> MLP --> tags
                          --> lookup table --> we_c --> biLSTM   --> we2_c


    TODO implement this option?
                          --> lookup table --> we     --> biLSTM  --> we2
    [word1, word2, ...]                                                       --> [we + we2_c] --> MLP --> tags
                          --> lookup table --> we_c   --> biLSTM  --> we2_c
    """

    def __init__(self, train_path, dev_path, test_path, log_frequency=1000, n_epochs=5, learning_rate=0.001):

        # for character-level embeddings
        characters = list("abcdefghijklmnopqrstuvwxyz ")
        characters.append(UNK_TOKEN)
        self.i2c = list(characters)
        self.c2i = {c: i for i, c in enumerate(characters)}
        self.nchars = len(characters)
        self.unk_c = self.c2i[UNK_TOKEN]

        POSTagger.__init__(self, train_path, dev_path, test_path, log_frequency, n_epochs, learning_rate)

    def build_model(self):
        """ This builds our POS-tagger model.
        """
        model = dy.ParameterCollection()

        params = {}
        word_embs_dim = 128
        params["E"] = model.add_lookup_parameters((self.n_words, word_embs_dim))

        # character-level embeddings
        input_size_c = 20
        output_size_c = 64  # hidden units
        params["CE"] = model.add_lookup_parameters((self.nchars, input_size_c))
        builders_c = [
            dy.LSTMBuilder(1, input_size_c, output_size_c, model),
            dy.LSTMBuilder(1, input_size_c, output_size_c, model)]

        # input encoder
        input_size = word_embs_dim + output_size_c * 2 # word_embedding size from lookup table + character embedding size
        output_size = 50  # hidden units
        builders = [
            dy.LSTMBuilder(1, input_size, output_size, model),
            dy.LSTMBuilder(1, input_size, output_size, model)]  # num layers, input size, hidden units, model

        params["H"] = model.add_parameters((32, output_size*2))
        params["O"] = model.add_parameters((self.n_tags, 32))

        self.model = model
        self.params = params
        self.builders = builders
        self.builders_c = builders_c

    def get_word_repr(self, w, add_noise=False):
        """
        """
        if isinstance(w, str):
            word = w
            word_id = self.word_vocab.w2i.get(w, self.unk)
        else:
            word_id = w
            word = self.word_vocab.i2w.get(w, self.unk)

        # get word_embedding
        w_emb = self.params["E"][word_id]

        # add char-level embedding
        char_ids = [self.c2i.get(c.lower(), self.unk_c) for c in word]
        char_embs = [self.params["CE"][cid] for cid in char_ids]

        f_init, b_init = [b.initial_state() for b in self.builders_c]

        fw_exps = f_init.transduce(char_embs)  # takes a list of expressions, feeds them and returns a list
        bw_exps = b_init.transduce(reversed(char_embs))
        c_emb = dy.concatenate([fw_exps[-1], bw_exps[-1]])

        # concatenate w_emb and c_emb
        emb = dy.concatenate([w_emb, c_emb])

        if add_noise:
            emb = dy.noise(emb, 0.1)  # Add gaussian noise to an expression (0.1 is the standard deviation of the gaussian)

        return emb


def main():

    # set up our data paths
    # data_dir = "/home/ubuntu/hd/home/lpmayos/code/datasets/ud2.1/ud-treebanks-v2.1/UD_English/"
    data_dir = "data/"
    train_path = os.path.join(data_dir, "en-ud-train.conllu")
    dev_path = os.path.join(data_dir, "en-ud-dev.conllu")
    test_path = os.path.join(data_dir, "en-ud-test.conllu")

    # create a POS tagger object
    pt = POSTagger2(train_path=train_path, dev_path=dev_path, test_path=test_path, n_epochs=3)
    pt.log_parameters()

    # let's train it!
    pt.train()

    test_accuracy = pt.evaluate(pt.test_data)
    logging.info("Test accuracy: {}".format(test_accuracy))


if __name__ == '__main__':

    start = time.process_time()
    main()
    logging.info("Elapsed time: {}".format(time.process_time() - start))

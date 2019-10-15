import os
import dynet as dy
import logging
from POSTagger import POSTagger, UNK_TOKEN


class POSTagger1(POSTagger):
    """
    POSTagger1 allows to use a character-based bi-LSTM for unknown words:

        [word1, word2, ...] --> lookup table or char_bilstm --> we --> biLSTM  --> MLP --> tags
    """

    def __init__(self, train_path, dev_path, test_path, log_frequency=1000, n_epochs=5, learning_rate=0.001, use_char_lstm=False):

        self.use_char_lstm = use_char_lstm

        if self.use_char_lstm:
            characters = list("abcdefghijklmnopqrstuvwxyz ")
            characters.append(UNK_TOKEN)
            self.i2c = list(characters)
            self.c2i = {c: i for i, c in enumerate(characters)}
            self.nchars = len(characters)
            self.unk_c = self.c2i[UNK_TOKEN]

        POSTagger.__init__(self, train_path, dev_path, test_path, log_frequency, n_epochs, learning_rate)

    def log_parameters(self):
        POSTagger.log_parameters(self)
        logging.info('use_char_lstm: % s' % self.use_char_lstm)
        logging.info('\n\n')

    def build_model(self):
        """ This builds our POS-tagger model.
        """
        model = dy.ParameterCollection()

        params = {"E": model.add_lookup_parameters((self.n_words, 128)),
                  "H": model.add_parameters((32, 50 * 2)),
                  "O": model.add_parameters((self.n_tags, 32))}

        builders = [
            dy.LSTMBuilder(1, 128, 50, model),
            dy.LSTMBuilder(1, 128, 50, model)]  # 1 layer, 128 input size, 50 hidden units, model

        if self.use_char_lstm:
            params["CE"] = model.add_lookup_parameters((self.nchars, 20))  # TODO how do I know if it is being updated un backward() <-- it should not, as I'm not using it in training (just for the missing words in test)
            self.fwdRNN_chars = dy.LSTMBuilder(1, 20, 64, model)
            self.bwdRNN_chars = dy.LSTMBuilder(1, 20, 64, model)

        self.model = model
        self.params = params
        self.builders = builders

    def get_word_repr(self, w, add_noise=False):
        """
        """
        if isinstance(w, str):
            word = w
            word_id = self.word_vocab.w2i.get(w, self.unk)
        else:
            word_id = w
            word = self.word_vocab.i2w.get(w, self.unk)

        if not self.use_char_lstm:
            emb = self.params["E"][word_id]
        else:
            if self.word_vocab.wc[w] > 1:  # TODO review min appearances to use embedding; originally 5
                emb = self.params["E"][word_id]
            else:
                char_ids = [self.c2i.get(c.lower(), self.unk_c) for c in word]
                char_embs = [self.params["CE"][cid] for cid in char_ids]

                f_init = self.fwdRNN_chars.initial_state()
                b_init = self.bwdRNN_chars.initial_state()

                fw_exps = f_init.transduce(char_embs)  # takes a list of expressions, feeds them and returns a list
                bw_exps = b_init.transduce(reversed(char_embs))
                emb = dy.concatenate([fw_exps[-1], bw_exps[-1]])

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
    pt = POSTagger1(train_path=train_path, dev_path=dev_path, test_path=test_path, n_epochs=1, use_char_lstm=False)
    pt.log_parameters()

    # let's train it!
    pt.train()

    test_accuracy = pt.evaluate(pt.test_data)
    logging.info("Test accuracy: {}".format(test_accuracy))


if __name__ == '__main__':
    main()

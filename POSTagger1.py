import os
import dynet as dy
import logging
from POSTaggerBase import POSTaggerBase, UNK_TOKEN


class POSTagger1(POSTaggerBase):
    """
    POSTagger1 allows to use an embeddings lookup table or a character-based bi-LSTM:

        [word1, word2, ...] --> lookup table or char_bilstm --> we --> biLSTM  --> MLP --> tags
    """

    def __init__(self, train_path, dev_path, test_path, log_path, log_frequency=1000, n_epochs=5, learning_rate=0.001, use_char_lstm=False):

        self.word_embs_dim = 128

        self.use_char_lstm = use_char_lstm

        # for character-level embeddings
        if self.use_char_lstm:
            characters = list("abcdefghijklmnopqrstuvwxyz ")
            characters.append(UNK_TOKEN)
            self.i2c = list(characters)
            self.c2i = {c: i for i, c in enumerate(characters)}
            self.nchars = len(characters)
            self.unk_c = self.c2i[UNK_TOKEN]

        POSTaggerBase.__init__(self, train_path, dev_path, test_path, log_path, log_frequency, n_epochs, learning_rate)

    def log_parameters(self):
        POSTaggerBase.log_parameters(self)
        logging.info('use_char_lstm: % s' % self.use_char_lstm)

    def build_model(self):
        """ This builds our POS-tagger model.
        """
        model = dy.ParameterCollection()

        params = {}

        if not self.use_char_lstm:
            word_embs_dim = self.word_embs_dim  # 128
            params["E"] = model.add_lookup_parameters((self.n_words, word_embs_dim))
        else:
            # character-level embeddings
            input_size_c = 20
            output_size_c = 64  # hidden units
            params["CE"] = model.add_lookup_parameters((self.nchars, input_size_c))
            builders_c = [
                dy.LSTMBuilder(1, input_size_c, output_size_c, model),
                dy.LSTMBuilder(1, input_size_c, output_size_c, model)]
            self.builders_c = builders_c

        # input encoder
        builders = [
            dy.LSTMBuilder(1, 128, 50, model),
            dy.LSTMBuilder(1, 128, 50, model)]  # 1 layer, 128 input size, 50 hidden units, model

        params["H"] = model.add_parameters((32, 50*2))
        params["O"] = model.add_parameters((self.n_tags, 32))

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
            char_ids = [self.c2i.get(c.lower(), self.unk_c) for c in word]
            char_embs = [self.params["CE"][cid] for cid in char_ids]

            f_init, b_init = [b.initial_state() for b in self.builders_c]

            fw_exps = f_init.transduce(char_embs)  # takes a list of expressions, feeds them and returns a list
            bw_exps = b_init.transduce(reversed(char_embs))
            emb = dy.concatenate([fw_exps[-1], bw_exps[-1]])

        if add_noise:
            emb = dy.noise(emb, 0.1)  # Add gaussian noise to an expression (0.1 is the standard deviation of the gaussian)

        return emb

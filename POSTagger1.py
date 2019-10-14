import dynet as dy
import logging
from POSTagger import POSTagger


class POSTagger1(POSTagger):
    """ A POS-tagger implemented in Dynet, based on https://github.com/clab/dynet/tree/master/examples/python

        # POS Tagger that allows to use a character-based bi-LSTM for unknown words

        # [word1, word2, ...] --> lookup table or char_bilstm --> we --> biLSTM  --> MLP --> tags
    """

    def __init__(self,
                 train_path="train.conll",
                 dev_path="dev.conll",
                 test_path="test.conll",
                 log_frequency=1000,
                 n_epochs=5,
                 learning_rate=0.001,
                 use_char_lstm=False):

        self.log_frequency = log_frequency
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.use_char_lstm = use_char_lstm

        # load data
        self.train_data, self.dev_data, self.test_data = POSTagger1.load_data(train_path, dev_path, test_path)

        # create vocabularies
        self.word_vocab, self.tag_vocab = self.create_vocabularies()

        self.unk = self.word_vocab.w2i[UNK_TOKEN]
        self.n_words = self.word_vocab.size()
        self.n_tags = self.tag_vocab.size()

        self.model, self.params, self.builders = self.build_model()

        self.log_parameters(train_path, dev_path, test_path)

    def log_parameters(self, train_path, dev_path, test_path):
        POSTagger.log_parameters(train_path, dev_path, test_path)
        logging.info('use_char_lstm: % s' % self.use_char_lstm)

    def build_model(self):
        """ This builds our POS-tagger model.
        """
        model = dy.ParameterCollection()

        params = {}
        params["E"] = model.add_lookup_parameters((self.n_words, 128))

        params["H"] = model.add_parameters((32, 50*2))
        params["O"] = model.add_parameters((self.n_tags, 32))

        builders = [
            dy.LSTMBuilder(1, 128, 50, model),
            dy.LSTMBuilder(1, 128, 50, model)]  # 1 layer, 128 input size, 50 hidden units, model

        if self.use_char_lstm:
            characters = list("abcdefghijklmnopqrstuvwxyz ")
            characters.append(UNK_TOKEN)
            self.i2c = list(characters)
            self.c2i = {c: i for i, c in enumerate(characters)}
            self.nchars = len(characters)
            self.unk_c = self.c2i[UNK_TOKEN]

            params["CE"] = model.add_lookup_parameters((self.nchars, 20))  # TODO how do I know if it is being updated un backward() <-- it should not, as I'm not using it in training (just for the missing words in test)
            self.fwdRNN_chars = dy.LSTMBuilder(1, 20, 64, model)
            self.bwdRNN_chars = dy.LSTMBuilder(1, 20, 64, model)

        return model, params, builders

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

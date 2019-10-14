import dynet as dy
from POSTagger import POSTagger


class POSTagger2(POSTagger):
    """
    # POS Tagger that concatenates word embeddings with character-level embeddings to represent words, and feeds them through a biLSTM to encode the words and generate tags

    # NOTICE that we need the char biLSTM because otherwise we_c would be an embedding for each character, and we need something with fixed size!
    # NOTICE how each sequence has a different length, but its OK, the LstmAcceptor doesn’t care. We create a new graph for each example, at exactly the desired length.
    #       --> in POSTagger4, we add mini-batching support


    #                       --> lookup table --> we
    # [word1, word2, ...]                                                         --> [we + we2_c] --> biLSTM --> MLP --> tags
    #                       --> lookup table --> we_c --> biLSTM   --> we2_c


    # TODO implement this option?
    #                       --> lookup table --> we     --> biLSTM  --> we2
    # [word1, word2, ...]                                                       --> [we + we2_c] --> MLP --> tags
    #                       --> lookup table --> we_c   --> biLSTM  --> we2_c

    """


    def __init__(self,
                 train_path="train.conll",
                 dev_path="dev.conll",
                 test_path="test.conll",
                 log_frequency=1000,
                 n_epochs=5,
                 learning_rate=0.001):
        """ Initialize the POS tagger.
        :param train_path: path to training data (CONLL format)
        :param dev_path: path to dev data (CONLL format)
        :param test_path: path to test data (CONLL format)
        """
        self.log_frequency = log_frequency
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate

        # load data
        self.train_data, self.dev_data, self.test_data = POSTagger2.load_data(train_path, dev_path, test_path)

        # create vocabularies
        self.word_vocab, self.tag_vocab = self.create_vocabularies()

        self.unk = self.word_vocab.w2i[UNK_TOKEN]
        self.n_words = self.word_vocab.size()
        self.n_tags = self.tag_vocab.size()

        # for character-level embeddings
        characters = list("abcdefghijklmnopqrstuvwxyz ")
        characters.append(UNK_TOKEN)
        self.i2c = list(characters)
        self.c2i = {c: i for i, c in enumerate(characters)}
        self.nchars = len(characters)
        self.unk_c = self.c2i[UNK_TOKEN]

        self.model, self.params, self.builders, self.builders_c = self.build_model()

        self.log_parameters(train_path, dev_path, test_path)


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

        return model, params, builders, builders_c

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

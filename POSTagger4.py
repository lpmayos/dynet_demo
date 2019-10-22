import dynet as dy
import logging
from POSTagger3 import POSTagger3

# NOTICE: We created a soft link to UniParse repository in cvt_text folder using ln -s ../UniParse/uniparse/ uniparse (lpmayos)
from uniparse.vocabulary import Vocabulary
from uniparse.parser_model import ParserModel
from uniparse.models.kiperwasser import DependencyParser


class POSTagger4(POSTagger3):
    """
    POSTagger4 concatenates word embeddings extracted from a graph-based parser with character-level embeddings to
    represent words, and feeds them through a biLSTM to encode the words and generate tags.
        More precisely, we parse the input sentence and for each word concatenate the internal states of the biLSTM
        TODO: learn weights
        TODO: try other combinations (avg, max...) <-- maybe in this task it does not matter ¿?

    we create a softlink to UniParse code:  ln -s ../UniParse/uniparse/ uniparse

    NOTICE that we need the char biLSTM because otherwise we_c would be an embedding for each character, and we need
    something with fixed size!
    NOTICE how each sequence has a different length, but its OK, the LstmAcceptor doesn’t care. We create a new graph
    for each example, at exactly the desired length.

                          --> K&G parser    --> we
    [word1, word2, ...]                                                       --> [we + we2_c] --> biLSTM --> MLP --> tags
                          --> lookup table  --> we_c --> biLSTM   --> we2_c

    """

    def __init__(self, train_path, dev_path, test_path, log_path, kg_vocab_path, kg_model_path, log_frequency=1000, n_epochs=5, learning_rate=0.001, batch_size=32):

        self.kg_vocab_path = kg_vocab_path
        self.kg_model_path = kg_model_path

        POSTagger3.__init__(self, train_path, dev_path, test_path, log_path, log_frequency, n_epochs, learning_rate, batch_size)

        # load K&G vocabulary

        only_words = True
        self.kg_vocab = Vocabulary(only_words)
        self.kg_vocab.load(self.kg_vocab_path)

        # create necessary parameters and initialize with K&G parameters values

        word_dim = 100  # copied from kiperwasser.py
        self.wlookup = self.model.add_lookup_parameters((self.kg_vocab.vocab_size, word_dim))

        upos_dim = 25  # copied from kiperwasser.py
        self.tlookup = self.model.add_lookup_parameters((self.kg_vocab.upos_size, upos_dim))

        bilstm_out = (word_dim + upos_dim) * 2  # 250
        self.deep_bilstm = dy.BiRNNBuilder(2, word_dim + upos_dim, bilstm_out, self.model, dy.VanillaLSTMBuilder)

        # populate parameters with K&G parameter values

        kiperwasser_model = DependencyParser(self.kg_vocab, None, False)
        contextual_embeddings_parser = ParserModel(kiperwasser_model, decoder="eisner", loss="kiperwasser", optimizer="adam", strategy="bucket", vocab=self.kg_vocab)
        contextual_embeddings_parser.load_from_file(self.kg_model_path)  # populates parameters

        # TODO how can I check if this is initializing parameters as expected?
        # TODO how can I check if ther are being trained later?
        self.deep_bilstm.param_collection().populate(self.kg_model_path, contextual_embeddings_parser._parser.deep_bilstm.param_collection().name())
        self.wlookup = contextual_embeddings_parser._parser.wlookup
        self.tlookup = contextual_embeddings_parser._parser.tlookup

    def log_parameters(self):
        POSTagger3.log_parameters(self)
        logging.info('kg_vocab_path: % s' % self.kg_vocab_path)
        logging.info('kg_model_path: % s' % self.kg_model_path)

    def extract_internal_states(self, word_ids, upos_ids):
        """ based on same function from UniParse.kiperwasser.py
        """
        n = word_ids.shape[-1]

        word_embs = [dy.lookup_batch(self.wlookup, word_ids[:, i]) for i in range(n)]
        upos_embs = [dy.lookup_batch(self.tlookup, upos_ids[:, i]) for i in range(n)]
        words = [dy.concatenate([w, p]) for w, p in zip(word_embs, upos_embs)]
        state_pairs_list = self.deep_bilstm.add_inputs(words)
        return state_pairs_list
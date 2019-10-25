import os
import dynet as dy
import numpy as np
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
        only_words = True
        self.kg_vocab = Vocabulary(only_words)
        self.kg_vocab.load(self.kg_vocab_path)

        POSTagger3.__init__(self, train_path, dev_path, test_path, log_path, log_frequency, n_epochs, learning_rate, batch_size)

    def save_needed_kg_parameters(self):
        """ Checks if it exists a partial model saving containing the parameters we need, and returns its path.
        It creates the partial model saving if it does not exist.
        """

        model_folder = os.path.dirname(os.path.abspath(self.kg_model_path)) + '/partial_params'
        if not os.path.exists(model_folder + '.meta'):

            # load K&G vocabulary and model
            parser = DependencyParser(self.kg_vocab, None, False)
            model = ParserModel(parser, decoder="eisner", loss="kiperwasser", optimizer="adam", strategy="bucket", vocab=self.kg_vocab)
            model.load_from_file(self.kg_model_path)  # populates parameters

            dy.save(model_folder, [parser.wlookup, parser.tlookup, parser.deep_bilstm])

        return model_folder

    def log_parameters(self):
        POSTagger3.log_parameters(self)
        logging.info('kg_vocab_path: % s' % self.kg_vocab_path)
        logging.info('kg_model_path: % s' % self.kg_model_path)

    def build_model(self):
        """ This builds our POS-tagger model.
        """
        model = dy.ParameterCollection()

        params = {}


        # load K&G vocabulary and model

        saved_params = self.save_needed_kg_parameters()
        self.wlookup, self.tlookup, self.deep_bilstm = dy.load(saved_params, model)
        word_embs_dim = self.deep_bilstm.builder_layers[0][0].spec[1] * 4

        # TODO how can I check if this is initializing parameters as expected?
        # TODO how can I check if they are being trained later?

        # character-level embeddings
        input_size_c = 20
        output_size_c = 64  # hidden units
        params["CE"] = model.add_lookup_parameters((self.nchars, input_size_c))
        builders_c = [
            dy.LSTMBuilder(1, input_size_c, output_size_c, model),
            dy.LSTMBuilder(1, input_size_c, output_size_c, model)]

        # input encoder
        input_size = word_embs_dim + output_size_c * 2  # word_embedding size from lookup table + character embedding size
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

    def extract_kg_internal_states(self, words_ids, tags_ids, format='concat'):
        """ based on same function from UniParse.kiperwasser.py
        TODO for now we return an array, but it may be a list to combine with weights, etc
        """

        # prepare data

        words = [self.word_vocab.i2w[w] for w in words_ids]
        tags = [self.tag_vocab.i2w[t] for t in tags_ids]

        input_data = self.kg_vocab.word_tags_tuple_to_conll(words, tags)
        words, lemmas, tags, heads, rels, chars = input_data[0]

        word_ids = np.array([words])
        tag_ids = np.array([tags])

        # extract internal states

        n = word_ids.shape[-1]

        word_embs = [dy.lookup_batch(self.wlookup, word_ids[:, i]) for i in range(n)]
        tag_embs = [dy.lookup_batch(self.tlookup, tag_ids[:, i]) for i in range(n)]
        words = [dy.concatenate([w, p]) for w, p in zip(word_embs, tag_embs)]
        state_pairs_list = self.deep_bilstm.add_inputs(words)

        return state_pairs_list

    def _get_contextual_repr_OLD(self, kg_internal_states):
        state_layer1 = kg_internal_states[0]
        hidden_state_layer1 = state_layer1.s()
        hidden_state_layer1_f = np.array(hidden_state_layer1[0].value())
        hidden_state_layer1_b = np.array(hidden_state_layer1[1].value())

        state_layer2 = kg_internal_states[1]
        hidden_state_layer2 = state_layer2.s()
        hidden_state_layer2_f = np.array(hidden_state_layer2[0].value())
        hidden_state_layer2_b = np.array(hidden_state_layer2[1].value())

        # TODO for now we just concatenate the expressions
        emb = np.concatenate((hidden_state_layer1_f, hidden_state_layer1_b, hidden_state_layer2_f, hidden_state_layer2_b))

        return emb

    def _get_contextual_repr(self, kg_internal_states):
        state_layer1 = kg_internal_states[0]
        hidden_state_layer1 = state_layer1.s()
        hidden_state_layer1_f = hidden_state_layer1[0]
        hidden_state_layer1_b = hidden_state_layer1[1]

        state_layer2 = kg_internal_states[1]
        hidden_state_layer2 = state_layer2.s()
        hidden_state_layer2_f = hidden_state_layer2[0]
        hidden_state_layer2_b = hidden_state_layer2[1]

        # TODO for now we just concatenate the expressions
        emb = dy.concatenate([hidden_state_layer1_f, hidden_state_layer1_b, hidden_state_layer2_f, hidden_state_layer2_b])

        return emb

    def get_word_repr(self, w, kg_internal_states, add_noise=False):
        """
        """
        if isinstance(w, str):
            word = w
            word_id = self.word_vocab.w2i.get(w, self.unk)
        else:
            word_id = w
            word = self.word_vocab.i2w.get(w, self.unk)

        # get contextual word_embedding
        w_emb = self._get_contextual_repr(kg_internal_states)

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

    def tag_sent(self, sent):
        """ Tags a single sentence.
        """
        dy.renew_cg()

        words, tags = sent
        kg_internal_states = self.extract_kg_internal_states(words, tags)

        f_init, b_init = [b.initial_state() for b in self.builders]

        wembs = [self.get_word_repr(w, kg_internal_states[i]) for i, w, t in enumerate(sent)]

        fw = [x.output() for x in f_init.add_inputs(wembs)]
        bw = [x.output() for x in b_init.add_inputs(reversed(wembs))]

        H = self.params["H"]
        O = self.params["O"]

        tags = []
        for f, b, (w, t) in zip(fw, reversed(bw), sent):
            # dense layer: dy.tanh(H * dy.concatenate([f, b]))
            # output layer: O (maps from H_dim --> output dimension)
            r_t = O * (dy.tanh(H * dy.concatenate([f, b])))
            out = dy.softmax(r_t)
            chosen = np.argmax(out.npvalue())
            tags.append(self.tag_vocab.i2w[chosen])

        return tags

    def build_tagging_graph(self, words, tags):
        """ Builds the graph for a single sentence.
        """

        kg_internal_states = self.extract_kg_internal_states(words, tags)

        f_init, b_init = [b.initial_state() for b in self.builders]

        wembs = [self.get_word_repr(w, kg_internal_states[i], add_noise=False) for i, w in enumerate(words)]  # TODO see what happens with/without adding noise

        # transduce takes a list of expressions, feeds them and returns a list; it is equivalent to:
        fw = f_init.transduce(wembs)            # fw = [x.output() for x in f_init.add_inputs(wembs)]
        bw = b_init.transduce(reversed(wembs))  # bw = [x.output() for x in b_init.add_inputs(reversed(wembs))]

        H = self.params["H"]
        O = self.params["O"]

        errs = []
        for f, b, t in zip(fw, reversed(bw), tags):
            f_b = dy.concatenate([f, b])
            r_t = O * (dy.tanh(H * f_b))
            err = dy.pickneglogsoftmax(r_t, t)  # equivalent to: dy.pick(-dy.log(dy.softmax(e1)), k)
            errs.append(err)
        return dy.esum(errs)
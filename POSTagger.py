# Source: https://github.com/neubig/lxmls-2017/blob/master/postagger.py

import dynet as dy
from collections import Counter
import random
import os
import numpy as np
import io
import logging

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s:%(levelname)s:\t%(message)s")

UNK_TOKEN = "_UNK_"
START_TOKEN = "_START_"


class POSTagger:
    """ A POS-tagger implemented in Dynet, based on https://github.com/clab/dynet/tree/master/examples/python
    """

    def __init__(self,
                 train_path="train.conll",
                 dev_path="dev.conll",
                 test_path="test.conll",
                 log_frequency=1000,
                 n_epochs=5,
                 learning_rate=0.001,
                 use_char_lstm=False):
        """ Initialize the POS tagger.
        :param train_path: path to training data (CONLL format)
        :param dev_path: path to dev data (CONLL format)
        :param test_path: path to test data (CONLL format)
        """
        self.log_frequency = log_frequency
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.use_char_lstm = use_char_lstm

        # load data
        self.train_data, self.dev_data, self.test_data = POSTagger.load_data(train_path, dev_path, test_path)

        # create vocabularies
        self.word_vocab, self.tag_vocab = self.create_vocabularies()

        self.unk = self.word_vocab.w2i[UNK_TOKEN]
        self.n_words = self.word_vocab.size()
        self.n_tags = self.tag_vocab.size()

        self.model, self.params, self.builders = self.build_model()

        self.log_parameters(train_path, dev_path, test_path)

    def log_parameters(self, train_path, dev_path, test_path):
        logging.info('log_frequency: %s' % self.log_frequency)
        logging.info('n_epochs: % s' % self.n_epochs)
        logging.info('learning_rate: % s' % self.learning_rate)
        logging.info('use_char_lstm: % s' % self.use_char_lstm)
        logging.info('train_path: % s' % train_path)
        logging.info('dev_path: % s' % dev_path)
        logging.info('test_path: % s' % test_path)
        logging.info('n_words: % s' % self.n_words)
        logging.info('n_tags: % s' % self.n_tags)

    @staticmethod
    def load_data(train_path=None, dev_path=None, test_path=None):
        """ Load all POS data.
        We store all data in memory so that we can shuffle easily each epoch.
        """
        train_data = list(read_conll_pos(train_path))
        dev_data = list(read_conll_pos(dev_path))
        test_data = list(read_conll_pos(test_path))
        return train_data, dev_data, test_data

    def create_vocabularies(self):
        """ Create vocabularies from the data.
        """
        words = []
        tags = []
        counter = Counter()
        for sentence in self.train_data:
            for word, tag in sentence:
                words.append(word)
                tags.append(tag)
                counter[word] += 1
        words.append(UNK_TOKEN)

        # replace frequency 1 words with unknown word token
        words = [w if counter[w] > 1 else UNK_TOKEN for w in words]

        tags.append(START_TOKEN)

        word_vocab = Vocab.from_corpus([words])
        tag_vocab = Vocab.from_corpus([tags])

        return word_vocab, tag_vocab

    def build_model(self):
        """ This builds our POS-tagger model.
        """
        model = dy.ParameterCollection()  # TODO old: dy.Model()

        params = {}
        params["E"] = model.add_lookup_parameters((self.n_words, 128))

        params["H"] = model.add_parameters((32, 50*2))
        params["O"] = model.add_parameters((self.n_tags, 32))

        builders = [
            dy.LSTMBuilder(1, 128, 50, model),
            dy.LSTMBuilder(1, 128, 50, model)]

        if self.use_char_lstm:
            characters = list("abcdefghijklmnopqrstuvwxyz ")
            characters.append(UNK_TOKEN)
            self.i2c = list(characters)
            self.c2i = {c: i for i, c in enumerate(characters)}
            self.nchars = len(characters)
            self.unk_c = self.c2i[UNK_TOKEN]

            params["CE"] = model.add_lookup_parameters((self.nchars, 20))
            self.fwdRNN_chars = dy.LSTMBuilder(1, 20, 64, model)
            self.bwdRNN_chars = dy.LSTMBuilder(1, 20, 64, model)

        return model, params, builders

    def word_repr(self, w):
        if not self.use_char_lstm:
            return self.params["E"][self.word_vocab.w2i.get(w, self.unk)]
        else:
            if self.word_vocab.wc[w] > 1:  # TODO review min appearances to use embedding; originally 5
                return self.params["E"][self.word_vocab.w2i.get(w, self.unk)]
            else:
                char_ids = [self.c2i.get(c.lower(), self.unk_c) for c in w]
                char_embs = [self.params["CE"][cid] for cid in char_ids]

                f_init = self.fwdRNN_chars.initial_state()
                b_init = self.bwdRNN_chars.initial_state()

                fw_exps = f_init.transduce(char_embs)  # takes a list of expressions, feeds them and returns a list
                bw_exps = b_init.transduce(reversed(char_embs))
                return dy.concatenate([fw_exps[-1], bw_exps[-1]])

    def tag_sent(self, sent, builders):
        """ Tags a single sentence.
        """
        dy.renew_cg()

        f_init, b_init = [b.initial_state() for b in builders]

        wembs = [self.word_repr(w) for w, t in sent]

        fw = [x.output() for x in f_init.add_inputs(wembs)]
        bw = [x.output() for x in b_init.add_inputs(reversed(wembs))]

        H = self.params["H"]
        O = self.params["O"]

        tags = []
        for f, b, (w, t) in zip(fw, reversed(bw), sent):
            r_t = O * (dy.tanh(H * dy.concatenate([f, b])))
            out = dy.softmax(r_t)
            chosen = np.argmax(out.npvalue())
            tags.append(self.tag_vocab.i2w[chosen])

        return tags

    def build_tagging_graph(self, words, tags, builders):
        """ Builds the graph for a single sentence.
        """
        dy.renew_cg()

        f_init, b_init = [b.initial_state() for b in builders]

        wembs = [self.params["E"][w] for w in words]
        # wembs = [dy.noise(we, 0.1) for we in wembs]  # TODO see what happens with/without adding noise

        # transduce takes a list of expressions, feeds them and returns a list; it is equivalent to:
        fw = f_init.transduce(wembs)            # fw = [x.output() for x in f_init.add_inputs(wembs)]
        bw = b_init.transduce(reversed(wembs))  # bw = [x.output() for x in b_init.add_inputs(reversed(wembs))]

        H = self.params["H"]
        O = self.params["O"]

        errs = []
        for f, b, t in zip(fw, reversed(bw), tags):
            f_b = dy.concatenate([f, b])
            r_t = O * (dy.tanh(H * f_b))
            err = dy.pickneglogsoftmax(r_t, t)
            errs.append(err)
        return dy.esum(errs)

    def evaluate(self, eval_data):
        """ Evaluate the model on the given data set.
        """
        good = bad = 0.0
        for sent in eval_data:
            tags = self.tag_sent(sent, self.builders)
            golds = [t for w, t in sent]
            for go, gu in zip(golds, tags):
                if go == gu:
                    good += 1
                else:
                    bad += 1
        accuracy = good / (good+bad)
        return accuracy

    def train(self):
        """ Training loop.
        """
        trainer = dy.AdamTrainer(self.model, alpha=self.learning_rate)
        tagged = 0
        loss = 0

        for EPOCH in range(self.n_epochs):
            random.shuffle(self.train_data)
            for i, s in enumerate(self.train_data, 1):

                # print loss
                if i % self.log_frequency == 0:
                    # trainer.status()
                    accuracy = self.evaluate(self.dev_data)
                    logging.info("Epoch {} Iter {} Loss: {:1.6f} Accuracy: {:1.4f}".format(
                        EPOCH, i, loss / tagged, accuracy))
                    loss = 0
                    tagged = 0

                # get loss for this training example
                words = [self.word_vocab.w2i.get(word, self.unk) for word, _ in s]
                tags = [self.tag_vocab.w2i[tag] for _, tag in s]

                sum_errs = self.build_tagging_graph(words, tags, self.builders)
                loss += sum_errs.scalar_value()
                tagged += len(tags)

                # update parameters
                sum_errs.backward()
                trainer.update()


class Vocab:
    def __init__(self, w2i, wc):
        self.w2i = dict(w2i)
        self.i2w = {i: w for w, i in w2i.items()}
        self.wc = wc

    @classmethod
    def from_corpus(cls, corpus):
        w2i = {}
        wc = Counter()
        for sent in corpus:
            for word in sent:
                w2i.setdefault(word, len(w2i))
                wc[word] += 1
        return Vocab(w2i, wc)

    def size(self):
        return len(self.w2i.keys())


def read_conll_pos(fname, word_column=1, tag_column=4):
    """ Read words and POS-tags from a CONLL file.
    """
    sent = []
    for line in io.open(fname, mode="r", encoding="utf-8"):
        if not line.startswith('#'):
            line = line.strip().split()
            if not line:
                if sent:
                    yield sent
                sent = []
            else:
                word = line[word_column]
                tag = line[tag_column]
                sent.append((word, tag))


def main():

    # set up our data paths
    data_dir = "/home/ubuntu/hd/home/lpmayos/code/datasets/ud2.1/ud-treebanks-v2.1/UD_English/"
    train_path = os.path.join(data_dir, "en-ud-train.conllu")
    dev_path = os.path.join(data_dir, "en-ud-dev.conllu")
    test_path = os.path.join(data_dir, "en-ud-test.conllu")
    use_char_lstm = False

    # create a POS tagger object
    pt = POSTagger(train_path=train_path, dev_path=dev_path, test_path=test_path, n_epochs=5, use_char_lstm=use_char_lstm)

    # let's train it!
    pt.train()

    test_accuracy = pt.evaluate(pt.test_data)
    logging.info("Test accuracy: {}".format(test_accuracy))


if __name__ == '__main__':
    main()
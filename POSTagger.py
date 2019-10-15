import sys
import dynet as dy
from collections import Counter
import random
import numpy as np
import io
import logging
import datetime

UNK_TOKEN = "_UNK_"
START_TOKEN = "_START_"


class POSTagger:
    """ A POS-tagger implemented in Dynet, based on https://github.com/clab/dynet/tree/master/examples/python
    """

    def __init__(self, train_path, dev_path, test_path, log_frequency=1000, n_epochs=5, learning_rate=0.001):

        self.config_logger()

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path

        self.log_frequency = log_frequency
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate

        # load data
        self.train_data, self.dev_data, self.test_data = self.load_data(train_path, dev_path, test_path)

        # create vocabularies
        self.word_vocab, self.tag_vocab = self.create_vocabularies()

        self.unk = self.word_vocab.w2i[UNK_TOKEN]
        self.n_words = self.word_vocab.size()
        self.n_tags = self.tag_vocab.size()

        self.build_model()

    def config_logger(self):
        log_filename = datetime.datetime.now().strftime("%Y%m%d.%H:%M.log")
        log_path = f'logs/{type(self).__name__}/{log_filename}'
        logging.basicConfig(level=logging.DEBUG,
                            filename=log_path,
                            format="%(asctime)s:%(levelname)s:\t%(message)s")
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    def log_parameters(self):
        logging.info('log_frequency: %s' % self.log_frequency)
        logging.info('n_epochs: % s' % self.n_epochs)
        logging.info('learning_rate: % s' % self.learning_rate)
        logging.info('train_path: % s' % self.train_path)
        logging.info('dev_path: % s' % self.dev_path)
        logging.info('test_path: % s' % self.test_path)
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
        raise NotImplementedError

    def get_word_repr(self, w, add_noise=False):
        """
        """
        raise NotImplementedError

    def tag_sent(self, sent):
        """ Tags a single sentence.
        """
        dy.renew_cg()

        f_init, b_init = [b.initial_state() for b in self.builders]

        wembs = [self.get_word_repr(w) for w, t in sent]

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

    def build_tagging_graph(self, words, tags):
        """ Builds the graph for a single sentence.
        """

        f_init, b_init = [b.initial_state() for b in self.builders]

        wembs = [self.get_word_repr(w, add_noise=False) for w in words]  # TODO see what happens with/without adding noise

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

    def train(self):
        """ Training loop without minibatching.
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

                dy.renew_cg()

                sum_errs = self.build_tagging_graph(words, tags)
                loss += sum_errs.scalar_value()
                tagged += len(tags)

                # update parameters
                sum_errs.backward()
                trainer.update()

    def evaluate(self, eval_data):
        """ Evaluate the model on the given data set.
        """
        good = bad = 0.0
        for sent in eval_data:
            tags = self.tag_sent(sent)
            golds = [t for w, t in sent]
            for go, gu in zip(golds, tags):
                if go == gu:
                    good += 1
                else:
                    bad += 1
        accuracy = good / (good+bad)
        return accuracy


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

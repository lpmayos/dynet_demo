import dynet as dy
import random
import logging
from POSTagger2 import POSTagger2


class POSTagger3(POSTagger2):
    """
    # Based on: https://github.com/neubig/lxmls-2017/blob/master/postagger.py and https://dynet.readthedocs.io/en/latest/minibatch.html

    # POS Tagger that concatenates word embeddings with character-level embeddings to represent words, and feeds them through a biLSTM to encode the words and generate tags

    #                       --> lookup table --> we
    # [word1, word2, ...]                                                         --> [we + we2_c] --> biLSTM --> MLP --> tags
    #                       --> lookup table --> we_c --> biLSTM   --> we2_c

    # NOTICE that we need the char biLSTM because otherwise we_c would be an embedding for each character, and we need something with fixed size!

    """

    def __init__(self,
                 train_path="train.conll",
                 dev_path="dev.conll",
                 test_path="test.conll",
                 log_frequency=1000,
                 n_epochs=5,
                 learning_rate=0.001,
                 batch_size=32):
        """ Initialize the POS tagger.
        :param train_path: path to training data (CONLL format)
        :param dev_path: path to dev data (CONLL format)
        :param test_path: path to test data (CONLL format)
        """

        self.log_frequency = log_frequency
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # load data
        self.train_data, self.dev_data, self.test_data = POSTagger3.load_data(train_path, dev_path, test_path)

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

    def log_parameters(self, train_path, dev_path, test_path):
        POSTagger.log_parameters(train_path, dev_path, test_path)
        logging.info('batch_size: % s' % self.batch_size)

    def _get_minibatches(self, n):
        """ returns batches of data of size self.batch_size
        """
        random.shuffle(self.train_data)
        l = len(self.train_data)
        for ndx in range(0, l, n):
            yield self.train_data[ndx:min(ndx + n, l)]

    def train(self):
        """ Training loop.
        """
        trainer = dy.AdamTrainer(self.model, alpha=self.learning_rate)
        tagged = 0
        loss = 0
        for epoch in range(self.n_epochs):
            i = 0
            for minibatch in self._get_minibatches(self.batch_size):
                dy.renew_cg()
                losses = []

                # print loss
                if i > 0 and i % self.log_frequency == 0:
                    trainer.status()
                    accuracy = self.evaluate(self.dev_data)
                    logging.info("Epoch {} Iter {} Loss: {:1.6f} Accuracy: {:1.4f}".format(epoch, i, loss / tagged, accuracy))
                    loss = 0
                    tagged = 0

                for sample in minibatch:

                    words = [self.word_vocab.w2i.get(word, self.unk) for word, _ in sample]
                    tags = [self.tag_vocab.w2i[tag] for _, tag in sample]
                    sample_loss = self.build_tagging_graph(words, tags)
                    losses.append(sample_loss)
                    i += 1
                    loss += sample_loss.scalar_value()
                    tagged += len(tags)

                minibatch_loss = dy.esum(losses)
                # minibatch_loss.forward()  # TODO is this necessary? I think it is done in build_tagging_graph
                minibatch_loss.backward()
                trainer.update()

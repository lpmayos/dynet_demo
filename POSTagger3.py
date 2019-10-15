import os
import dynet as dy
import random
import time
import logging
from POSTagger2 import POSTagger2


class POSTagger3(POSTagger2):
    """
    POSTagger3 concatenates word embeddings with character-level embeddings to represent words, and feeds them through a
    biLSTM to encode the words and generate tags. It allows minibatching.

                          --> lookup table --> we
    [word1, word2, ...]                                                       --> [we + we2_c] --> biLSTM --> MLP --> tags
                          --> lookup table --> we_c --> biLSTM   --> we2_c

    NOTICE that we need the char biLSTM because otherwise we_c would be an embedding for each character, and we need
    something with fixed size!
    """

    def __init__(self, train_path, dev_path, test_path, log_frequency=1000, n_epochs=5, learning_rate=0.001, batch_size=32):

        self.batch_size = batch_size

        POSTagger2.__init__(self, train_path, dev_path, test_path, log_frequency, n_epochs, learning_rate)

    def log_parameters(self):
        POSTagger2.log_parameters(self)
        logging.info('batch_size: % s' % self.batch_size)
        logging.info('\n\n')

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


def main():

    # set up our data paths
    # data_dir = "/home/ubuntu/hd/home/lpmayos/code/datasets/ud2.1/ud-treebanks-v2.1/UD_English/"
    data_dir = "data/"
    train_path = os.path.join(data_dir, "en-ud-train.conllu")
    dev_path = os.path.join(data_dir, "en-ud-dev.conllu")
    test_path = os.path.join(data_dir, "en-ud-test.conllu")

    batch_size = 32
    log_frequency = round(1000 / batch_size)

    # create a POS tagger object
    pt = POSTagger3(train_path=train_path, dev_path=dev_path, test_path=test_path, log_frequency=log_frequency, n_epochs=3, batch_size=batch_size)
    pt.log_parameters()

    # let's train it!
    pt.train()

    test_accuracy = pt.evaluate(pt.test_data)
    logging.info("Test accuracy: {}".format(test_accuracy))


if __name__ == '__main__':

    start = time.process_time()
    main()
    logging.info("Elapsed time: {}".format(time.process_time() - start))


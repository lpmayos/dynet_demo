import logging
import os

from POSTagger1 import POSTagger1
from POSTagger2 import POSTagger2
from POSTagger3 import POSTagger3


def main(config):

    # set up our data paths
    # data_dir = "/home/ubuntu/hd/home/lpmayos/code/datasets/ud2.1/ud-treebanks-v2.1/UD_English/"
    data_dir = "data/"
    train_path = os.path.join(data_dir, "en-ud-train.conllu")
    dev_path = os.path.join(data_dir, "en-ud-dev.conllu")
    test_path = os.path.join(data_dir, "en-ud-test.conllu")

    # create a POS tagger object
    n_epochs = 1

    if config['tagger_version'] == 1:
        use_char_lstm = config['tagger_1_config']['use_char_lstm']
        pt = POSTagger1(train_path=config['train_path'], dev_path=config['dev_path'], test_path=config['test_path'], n_epochs=config['n_epochs'], use_char_lstm=use_char_lstm)

    elif config['tagger_version'] == 2:
        pt = POSTagger2(train_path=config['train_path'], dev_path=config['dev_path'], test_path=config['test_path'], n_epochs=config['n_epochs'])

    elif config['tagger_version'] == 3:
        batch_size = config['tagger_3_config']['batch_size']
        pt = POSTagger3(train_path=config['train_path'], dev_path=config['dev_path'], test_path=config['test_path'], n_epochs=config['n_epochs'], batch_size=batch_size)

    pt.log_parameters()

    # let's train it!
    if config['train']:
        # TODO save model https://dynet.readthedocs.io/en/latest/python_saving_tutorial.html
        pt.train()
    else:
        #TODO load model
        print('')

    test_accuracy = pt.evaluate(pt.test_data)  # TODO save results
    logging.info("Test accuracy: {}".format(test_accuracy))


if __name__ == '__main__':

    data_dir = "data/"  # "/home/ubuntu/hd/home/lpmayos/code/datasets/ud2.1/ud-treebanks-v2.1/UD_English/"
    train_path = os.path.join(data_dir, "en-ud-train.conllu")
    dev_path = os.path.join(data_dir, "en-ud-dev.conllu")
    test_path = os.path.join(data_dir, "en-ud-test.conllu")

    config = {
        'tagger_version': 1,
        'train_path': os.path.join(data_dir, "en-ud-train.conllu"),
        'dev_path': os.path.join(data_dir, "en-ud-dev.conllu"),
        'test_path': os.path.join(data_dir, "en-ud-test.conllu"),
        'train': True,
        'model_path': 'model1',
        'tagger_1_config': { 'use_char_lstm': False},
        'tagger_3_config': {'batch_size': 32}
    }

    main(config)
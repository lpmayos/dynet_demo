import argparse
import json
import logging
import os
import time

from POSTagger1 import POSTagger1
from POSTagger2 import POSTagger2
from POSTagger3 import POSTagger3


def main(config):

    # set up our data paths
    train_path = config['data_dir'] + config['train_file']
    dev_path = config['data_dir'] + config['dev_file']
    test_path = config['data_dir'] + config['test_file']

    log_path = config['model_dir'] + '/log.log'
    model_path = config['model_dir'] + '/model.model'


    # create a POS tagger object

    if config['tagger_version'] == 1:
        use_char_lstm = config['use_char_lstm']
        pt = POSTagger1(train_path=train_path, dev_path=dev_path, test_path=test_path, log_path=log_path, n_epochs=config['n_epochs'], use_char_lstm=use_char_lstm)

    elif config['tagger_version'] == 2:
        pt = POSTagger2(train_path=train_path, dev_path=dev_path, test_path=test_path, log_path=log_path, n_epochs=config['n_epochs'])

    elif config['tagger_version'] == 3:
        batch_size = config['batch_size']
        pt = POSTagger3(train_path=train_path, dev_path=dev_path, test_path=test_path, log_path=log_path, n_epochs=config['n_epochs'], batch_size=batch_size)

    logging.info('----------------------------------------------------------------------------------------------------')

    pt.log_parameters()

    # let's train it!
    if config['train']:
        pt.train()
        pt.save_model(model_path)  # info: https://dynet.readthedocs.io/en/latest/python_saving_tutorial.html
    else:
        pt.load_model(model_path)

    test_accuracy = pt.evaluate(pt.test_data)  # TODO save results
    logging.info('')
    logging.info("Test accuracy: {}".format(test_accuracy))

    logging.info('----------------------------------------------------------------------------------------------------\n\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--config_file", dest="config_file", help="JSON file containing tge configuration to be used", required=True)

    arguments, unknown = parser.parse_known_args()

    with open(arguments.config_file, encoding="UTF-8") as config_file:
        config = json.load(config_file)
        # we'll save the model data in the same folder as configiguration json file
        config['model_dir'] = os.path.dirname(config_file.name)

    start = time.process_time()
    main(config)
    logging.info("Elapsed time: {}".format(time.process_time() - start))

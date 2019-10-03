from collections import Counter

import dynet as dy
import numpy as np
import random


# data

def read_input(fname):
    # format of files: each line is "word<TAB>tag<newline>", blank line is new sentence.
    sent = []
    for line in open(fname):
        line = line.strip().split()
        if not line:
            if sent: yield sent
            sent = []
        else:
            w,p = line
    sent.append((w, p))


train_file = "data/train.txt"
dev_file = "data/dev.txt"
test_file = "data/test.txt"

train = list(read_input(train_file))
dev = list(read_input(dev_file))
test = list(read_input(test_file))

words = []
tags = []
wc = Counter()
for s in train:
    for w, p in s:
        words.append(w)
        tags.append(p)
        wc[w] += 1
words.append("_UNK_")
tags.append("_START_")

for s in dev:
    for w, p in s:
        words.append(w)

vw = util.Vocab.from_corpus([words])
vt = util.Vocab.from_corpus([tags])
UNK = vw.w2i["_UNK_"]

nwords = vw.size()
ntags = vt.size()

# parameters

params = dy.ParameterCollection()

nwords = xxx
ntags = xxx
num_layers = 1
input_dims = 128
output_dims = 50

WORDS_LOOKUP = params.add_lookup_parameters((nwords, input_dims))

fwdRNN = dy.LSTMBuilder(num_layers, input_dims, output_dims, params)
bwdRNN = dy.LSTMBuilder(num_layers, input_dims, output_dims, params)

H = params.add_parameters((32, 50*2))
O = params.add_parameters((ntags, 32))

# if we want to use a chars LSTM:

nchars = xxx
input_dims_chars = 20
output_dims_chars = 64
CHARS_LOOKUP = params.add_lookup_parameters((nchars, input_dims_chars))

fwdRNN_chars = dy.LSTMBuilder(num_layers, input_dims_chars, output_dims_chars, params)
bwdRNN_chars = dy.LSTMBuilder(num_layers, input_dims_chars, output_dims_chars, params)

trainer = dy.SimpleSGDTrainer(params)


def build_tagging_graph(words):
    dy.renew_cg()

    fw_init = fwdRNN.initial_state()
    bw_init = bwdRNN.initial_state()

    wembs = [word_rep(w) for w in words]

    # transduce is equivalent to:
    # fw_exps = []
    # s = f_init
    # for we in wembs:
    #     s = s.add_input(we)
    #     fw_exps.append(s.output())
    fw_exps = fw_init.transduce(wembs)  # takes a list of expressions, feeds them and returns a list
    bw_exps = bw_init.transduce(reversed(wembs))

    bi = [dy.concatenate([f, b]) for f,b in zip(fw_exps, reversed(bw_exps))]

    outs = [O * (dy.tanh(H*x)) for x in bi]
    return outs


def word_rep(w, use_char_lstm=False):
    if not use_char_lstm or wc[w] > 5:
        w_index = vw.w2i[w]
        return WORDS_LOOKUP[w_index]
    else:
        char_ids = [vc.w2i[c] for c in w]
        char_embs = [CHARS_LOOKUP[cid] for cid in char_ids]
        fw_exps = fwdRNN_chars.transduce(char_embs)  # takes a list of expressions, feeds them and returns a list
        bw_exps = bwdRNN_chars.transduce(reversed(char_embs))
        return dy.concatenate([fw_exps[-1], bw_exps[-1]])


def tag_sent(words):
    vecs = build_tagging_graph(words)

    vecs = [dy.softmax(v) for v in vecs]
    probs = [v.npvalue() for v in vecs]

    tags = []
    for prb in probs:
        tag = np.argmax(prb)
        tags.append(vt.i2w[tag])

    return zip(words, tags)


def sent_loss(words, tags):
    vecs = build_tagging_graph(words)

    losses = []
    for v, t in zip(vecs, tags):
        tid = vt.w2i[t]
        loss = dy.pickneglogsoftmax(v, tid)
        losses.append(loss)

    return dy.esum(losses)


loss = 0
tagged = 0
for ITER in xrange(50):
    random.shuffle(train)
    for i, s in enumerate(train, 1):

        # progress reports

        if i > 0 and i % 500 == 0:
            trainer.status()
            print(loss / tagged)
            loss = 0
            tagged = 0

        if i % 10000 == 0:
            good = bad = 0.0
            for sent in dev:
                words = [w for w, t in sent]
                golds = [t for w, t in sent]
                tags = [t for w, t in tag_sent(words)]
                for go, gu in zip(golds, tags):
                    if go == gu:
                        good += 1
                    else:
                        bad += 1
            print(good/(good+bad))

        words = [w for w, t in s]
        golds = [t for w, t in s]

        loss_exp = sent_loss(words, golds)
        loss += loss_exp.scalar_value()
        tagged += len(golds)
        loss_exp.backward()
        trainer.update()

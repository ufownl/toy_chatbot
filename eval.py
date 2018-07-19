import math
import random
import mxnet as mx
from dataset import load_conversations, dataset_filter, make_vocab, tokenize, pad_sentence
from seq2seq_lstm import Seq2seqLSTM

context = mx.cpu()
num_embed = 128
num_hidden = 1024
num_layers = 2
sequence_length = 64
sample_size = 1024

print("Loading dataset...", flush=True)
dataset = dataset_filter(load_conversations("data/xiaohuangji50w_nofenci.conv"), sequence_length)
vocab = make_vocab(dataset)
dataset = tokenize(dataset, vocab)

print("Loading model...", flush=True)
model = Seq2seqLSTM(vocab.size(), num_embed, num_hidden, num_layers)
model.load_parameters("model/seq2seq_lstm.params", ctx=context)

print("Evaluating...", flush=True)
ppl = mx.metric.Perplexity(ignore_label=None)
for source, target in random.sample(dataset, sample_size):
    source = pad_sentence(source, vocab, [2 ** (i + 1) for i in range(int(math.log(sequence_length, 2)))])
    source = mx.nd.reverse(mx.nd.array(source, ctx=context), axis=0)
    label = mx.nd.array(target + [vocab.char2idx("<EOS>")], ctx=context)
    target = mx.nd.array([vocab.char2idx("<GO>")] + target, ctx=context)
    hidden = model.begin_state(func=mx.nd.zeros, batch_size=1, ctx=context)
    output, hidden = model(source.reshape((1, -1)).T, target.reshape((1, -1)).T, hidden)
    probs = mx.nd.softmax(output, axis=1)
    ppl.update([label], [probs])
print(ppl.get())

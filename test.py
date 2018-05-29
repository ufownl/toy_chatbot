import mxnet as mx
from dataset import load_conversations, dataset_filter, make_vocab
from seq2seq_lstm import Seq2seqLSTM

context = mx.cpu()
num_embed = 128
num_hidden = 1024
num_layers = 2
sequence_length = 16

print("Loading dataset...", flush=True)
dataset = dataset_filter(load_conversations("data/xiaohuangji50w_nofenci.conv"), sequence_length)
vocab = make_vocab(dataset)

print("Loading model...", flush=True)
model = Seq2seqLSTM(vocab.size(), num_embed, num_hidden, num_layers)
model.load_params("model/seq2seq_lstm.params", ctx=context)

while True:
    try:
        source = input("> ")
    except EOFError:
        print("")
        break;
    source = [vocab.char2idx(ch) for ch in source]
    if sequence_length > len(source):
        source += [vocab.char2idx("<PAD>")] * (sequence_length - len(source))
    #print(source)
    source = mx.nd.array(source, ctx=context)
    hidden = model.begin_state(func=mx.nd.zeros, batch_size=1, ctx=context)
    hidden = model.encode(source.reshape((1, -1)).T, hidden)
    target = mx.nd.array([vocab.char2idx("<GO>")], ctx=context)
    while True:
        output, hidden = model.decode(target.reshape((1, -1)).T, hidden)
        probs = mx.nd.softmax(output, axis=1)
        index = mx.nd.random.multinomial(probs)
        if index[-1].asscalar() == vocab.char2idx("<EOS>"):
            break;
        target = mx.nd.array([index[-1].asscalar()], ctx=context)
        print(vocab.idx2char(index[-1].asscalar()), end="", flush=True)
    print("") 

import math
import mxnet as mx
from dataset import load_conversations, dataset_filter, make_vocab, pad_sentence
from seq2seq_lstm import Seq2seqLSTM

context = mx.cpu()
num_embed = 128
num_hidden = 1024
num_layers = 2
sequence_length = 64
beam_size = 10

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
    source = pad_sentence(source, vocab, [2 ** (i + 1) for i in range(int(math.log(sequence_length, 2)))])
    print(source)
    source = mx.nd.reverse(mx.nd.array(source, ctx=context), axis=0)
    hidden = model.begin_state(func=mx.nd.zeros, batch_size=1, ctx=context)
    hidden = model.encode(source.reshape((1, -1)).T, hidden)
    sequences = [([vocab.char2idx("<GO>")], 0.0, hidden)]
    while True:
        candidates = []
        for seq, score, hidden in sequences:
            if seq[-1] == vocab.char2idx("<EOS>"):
                candidates.append((seq, score, hidden))
            else:
                target = mx.nd.array([seq[-1]], ctx=context)
                output, hidden = model.decode(target.reshape((1, -1)).T, hidden)
                probs = mx.nd.softmax(output, axis=1)
                beam = probs.reshape((-1,)).topk(k=beam_size, ret_typ="both")
                for i in range(beam_size):
                    candidates.append((seq + [int(beam[1][i].asscalar())], score + math.log(beam[0][i].asscalar()), hidden))
        if len(candidates) <= len(sequences):
            break;
        sequences = sorted(candidates, key=lambda tup: tup[1], reverse=True)[:beam_size]

    scores = mx.nd.array([score for _, score, _ in sequences], ctx=context)
    probs = mx.nd.softmax(scores)

    for i, (seq, score, _) in enumerate(sequences):
        text = ""
        for token in seq[1:-1]:
            text += vocab.idx2char(token)
        print(text, score, probs[i].asscalar())

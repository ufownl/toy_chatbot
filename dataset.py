import jieba
import mxnet as mx
from vocab import Vocabulary

def load_conversations(path):
    with open(path, "r") as f:
        raw = f.read()

    dataset = []
    for conv in raw.split("E\n")[1:]:
        qa = conv.split("\n")
        if len(qa) >= 2:
            dataset.append((list(jieba.cut(qa[0][2:])), list(jieba.cut(qa[1][2:]))))

    return dataset


def dataset_filter(dataset, sequence_length):
    return [(src, tgt) for src, tgt in dataset if len(src) <= sequence_length and len(tgt) <= sequence_length]


def make_vocab(dataset, max_size=8192):
    freq = {}
    words = [w for conv in dataset for sent in conv for w in sent]
    for w in words:
        if w in freq:
            freq[w] += 1
        else:
            freq[w] = 1
    freq = sorted(freq.items(), key=lambda tup: tup[1], reverse=True)
    return Vocabulary([k for k, _ in freq[:max_size]])


def tokenize(dataset, vocab):
    return [tuple([vocab.word2idx(w) for w in sent] for sent in conv) for conv in dataset]


def rnn_buckets(dataset, buckets):
    min_src_len = -1
    for max_src_len in buckets:
        src_bucket = [(src, tgt) for src, tgt in dataset if len(src) > min_src_len and len(src) <= max_src_len]
        min_src_len = max_src_len
        if len(src_bucket) > 0:
            min_tgt_len = -1
            for max_tgt_len in buckets:
                tgt_bucket = [(src, tgt) for src, tgt in src_bucket if len(tgt) > min_tgt_len and len(tgt) <= max_tgt_len]
                min_tgt_len = max_tgt_len
                if len(tgt_bucket) > 0:
                    yield tgt_bucket, max_src_len, max_tgt_len


def rnn_batches(dataset, vocab, batch_size, source_length, target_length, ctx):
    src_tok, tgt_tok = zip(*dataset)
    src_tok, tgt_tok = list(src_tok), list(tgt_tok)
    batches = len(dataset) // batch_size
    if batches * batch_size < len(dataset):
        batches += 1
    for i in range(0, batches):
        start = i * batch_size
        src_bat = mx.nd.array(_pad_batch(src_tok[start: start + batch_size], vocab, source_length), ctx=ctx)
        src_bat = mx.nd.reverse(src_bat, axis=1)
        tgt_bat = mx.nd.array(_pad_batch(_add_sent_prefix(tgt_tok[start: start + batch_size], vocab), vocab, target_length + 1), ctx=ctx)
        lbl_bat = mx.nd.array(_pad_batch(_add_sent_suffix(tgt_tok[start: start + batch_size], vocab), vocab, target_length + 1), ctx=ctx)
        yield src_bat.T, tgt_bat.T, lbl_bat.T.reshape((-1,))


def pad_sentence(sent, vocab, buckets):
    min_len = -1
    for max_len in buckets:
        if len(sent) > min_len and len(sent) <= max_len:
            return sent + [vocab.word2idx("<PAD>")] * (max_len - len(sent))
        min_len = max_len
    return sent


def _add_sent_prefix(batch, vocab):
    return [[vocab.word2idx("<GO>")] + sent for sent in batch]


def _add_sent_suffix(batch, vocab):
    return [sent + [vocab.word2idx("<EOS>")] for sent in batch]


def _pad_batch(batch, vocab, seq_len):
    return [sent + [vocab.word2idx("<PAD>")] * (seq_len - len(sent)) for sent in batch]


if __name__ == "__main__":
    dataset = load_conversations("data/xiaohuangji50w_nofenci.conv")
    print("dataset size: ", len(dataset))
    dataset = dataset_filter(dataset, 64)
    print("filtered dataset size: ", len(dataset))
    print("dataset preview: ", dataset[:10])
    vocab = make_vocab(dataset)
    print("vocab size: ", vocab.size())
    dataset = tokenize(dataset, vocab)
    print("tokenize dataset preview: ", dataset[:10])
    print("buckets preview: ", [(len(bucket), (src_len, tgt_len)) for bucket, src_len, tgt_len in rnn_buckets(dataset, [2, 4, 8, 16, 32, 64])])
    print("batch preview: ", next(rnn_batches(dataset, vocab, 4, 64, 64, mx.cpu())))

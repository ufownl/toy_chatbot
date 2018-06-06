import os
import time
import math
import random
import mxnet as mx
from dataset import load_conversations, dataset_filter, make_vocab, tokenize, rnn_buckets, rnn_batches
from seq2seq_lstm import Seq2seqLSTM

def main(num_embed, num_hidden, num_layers, batch_size, sequence_length, context):
    print("Loading dataset...", flush=True)
    dataset = dataset_filter(load_conversations("data/xiaohuangji50w_nofenci.conv"), sequence_length)
    vocab = make_vocab(dataset)
    dataset = tokenize(dataset, vocab)

    model = Seq2seqLSTM(vocab.size(), num_embed, num_hidden, num_layers)
    loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()

    if os.path.isfile("model/seq2seq_lstm.ckpt"):
        with open("model/seq2seq_lstm.ckpt", "r") as f:
            ckpt_lines = f.readlines()
        ckpt_argv = ckpt_lines[-1].split()
        epoch = int(ckpt_argv[0])
        best_L = float(ckpt_argv[1])
        learning_rate = float(ckpt_argv[2])
        epochs_no_progress = int(ckpt_argv[3])
        model.load_params("model/seq2seq_lstm.params", ctx=context)
    else:
        epoch = 0
        best_L = float("Inf")
        epochs_no_progress = 0
        learning_rate = 0.001
        model.initialize(mx.init.Xavier(), ctx=context)

    print("Learning rate:", learning_rate)
    print("Training...", flush=True)
    trainer = mx.gluon.Trainer(model.collect_params(), "Adam",
                               {"learning_rate": learning_rate, "clip_gradient": 5.0})
    while learning_rate >= 1e-8:
        random.shuffle(dataset)
        ts = time.time()
        total_L = 0.0
        batch = 0
        for bucket, max_len in rnn_buckets(dataset, [2 ** (i + 1) for i in range(int(math.log(sequence_length, 2)))]):
            for source, target, label in rnn_batches(bucket, vocab, batch_size, sequence_length, max_len, context):
                batch += 1
                hidden = model.begin_state(func=mx.nd.zeros, batch_size=batch_size, ctx=context)
                with mx.autograd.record():
                    output, hidden = model(source, target, hidden)
                    L = loss(output, label)
                    L.backward()
                trainer.step(batch_size)
                batch_L = mx.nd.mean(L).asscalar()
                if batch_L != batch_L:
                    raise ValueError()
                total_L += batch_L
                print("[Epoch %d  Bucket %d  Batch %d]  batch_loss %.10f  average_loss %.10f  elapsed %.2fs" %
                    (epoch, max_len, batch, batch_L, total_L / batch, time.time() - ts), flush=True)
        epoch += 1

        avg_L = total_L / batch
        print("[Epoch %d]  learning_rate %.10f  loss %.10f  epochs_no_progress %d  duration %.2fs" %
            (epoch, learning_rate, avg_L, epochs_no_progress, time.time() - ts), flush=True)

        if avg_L < best_L:
            best_L = avg_L
            epochs_no_progress = 0
            model.save_params("model/seq2seq_lstm.params")
            with open("model/seq2seq_lstm.ckpt", "a") as f:
                f.write("%d %.10f %.10f %d\n" % (epoch, best_L, learning_rate, epochs_no_progress))
        elif epochs_no_progress < 2:
            epochs_no_progress += 1
        else:
            epochs_no_progress = 0
            learning_rate *= 0.5
            trainer.set_learning_rate(learning_rate)


if __name__ == "__main__":
    while True:
        try:
            main(num_embed=128, num_hidden=1024, num_layers=2, batch_size=256, sequence_length=64, context=mx.gpu())
            break;
        except ValueError:
            print("Oops! The value of loss become NaN...")

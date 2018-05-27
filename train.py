import os
import time
import random
import mxnet as mx
from dataset import load_conversations, dataset_filter, make_vocab, tokenize, rnn_batches
from seq2seq_lstm import Seq2seqLSTM

def main(num_embed, num_hidden, num_layers, gradient_clip, batch_size, sequence_length, context):
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

    print("Training...", flush=True)
    trainer = mx.gluon.Trainer(model.collect_params(), "adam", {"learning_rate": learning_rate})
    while learning_rate >= 1e-5:
        random.shuffle(dataset)
        ts = time.time()
        total_L = 0.0
        for i, (source, target, label) in enumerate(rnn_batches(dataset, vocab, batch_size, sequence_length, context)):
            hidden = model.begin_state(func=mx.nd.zeros, batch_size=batch_size, ctx=context)
            with mx.autograd.record():
                output, hidden = model(source, target, hidden)
                L = loss(output, label)
                L.backward()
            grads = [p.grad(context) for p in model.collect_params().values()]
            mx.gluon.utils.clip_global_norm(grads, gradient_clip)
            trainer.step(batch_size)
            batch_L = mx.nd.mean(L).asscalar()
            if batch_L != batch_L:
                raise ValueError()
            total_L += batch_L
            if (i + 1) % 100 == 0:
                print("[Epoch %d  Batch %d]  loss %f  elapsed %.2fs" %
                    (epoch, i + 1, total_L / (i + 1), time.time() - ts), flush=True)
        epoch += 1

        avg_L = total_L / (len(dataset) // batch_size)
        print("[Epoch %d]  learning_rate %f  loss %f  epochs_no_progress %d  duration %.2fs" %
            (epoch, learning_rate, avg_L, epochs_no_progress, time.time() - ts), flush=True)

        if avg_L < best_L:
            best_L = avg_L
            epochs_no_progress = 0
            model.save_params("model/seq2seq_lstm.params")
            with open("model/seq2seq_lstm.ckpt", "a") as f:
                f.write("%d %f %f %d\n" % (epoch, best_L, learning_rate, epochs_no_progress))
        elif epochs_no_progress < 2:
            epochs_no_progress += 1
        else:
            epochs_no_progress = 0
            learning_rate *= 0.5
            trainer.set_learning_rate(learning_rate)


if __name__ == "__main__":
    while True:
        try:
            main(num_embed=128, num_hidden=1024, num_layers=2, gradient_clip=5, batch_size=1024, sequence_length=16, context=mx.gpu())
            break;
        except ValueError:
            print("Oops! The value of loss become NaN...")

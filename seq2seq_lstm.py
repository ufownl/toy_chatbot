import mxnet as mx

class Seq2seqLSTM(mx.gluon.Block):
    def __init__(self, vocab_size, num_embed, num_hidden, num_layers, dropout=0.5, **kwargs):
        super(Seq2seqLSTM, self).__init__(**kwargs)
        with self.name_scope():
            self._embed = mx.gluon.nn.Embedding(vocab_size, num_embed, weight_initializer=mx.init.Uniform(0.1))
            self._encode = mx.gluon.rnn.LSTM(num_hidden, num_layers)
            self._decode = mx.gluon.rnn.LSTM(num_hidden, num_layers)
            self._dropout = mx.gluon.nn.Dropout(dropout)
            self._output = mx.gluon.nn.Dense(vocab_size)
        self._num_hidden = num_hidden

    def forward(self, source, target, hidden):
        hidden = self.encode(source, hidden)
        return self.decode(target, hidden)

    def encode(self, inputs, hidden):
        embed = self._embed(inputs)
        _, hidden = self._encode(embed, hidden)
        return hidden

    def decode(self, inputs, hidden):
        embed = self._embed(inputs)
        output, hidden = self._decode(embed, hidden)
        output = self._dropout(output)
        output = self._output(output.reshape((-1, self._num_hidden)))
        return output, hidden

    def begin_state(self, *args, **kwargs):
        return self._encode.begin_state(*args, **kwargs)

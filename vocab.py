import json

class Vocabulary:
    def __init__(self, words=None):
        if words:
            self._words = ["<UNK>", "<PAD>", "<GO>", "<EOS>"] + words
            self._word_indices = dict((c, i) for i, c in enumerate(self._words))
            self._indices_word = dict((i, c) for i, c in enumerate(self._words))

    def size(self):
        return len(self._words)

    def word2idx(self, ch):
        if ch not in self._word_indices:
            ch = "<UNK>"
        return self._word_indices[ch]

    def idx2word(self, idx):
        return self._indices_word[idx]

    def save(self, path):
        with open(path, "w") as f:
            f.write(json.dumps(self._words))

    def load(self, path):
        with open(path, "r") as f:
            s = f.read()
        self._words = json.loads(s)
        self._word_indices = dict((c, i) for i, c in enumerate(self._words))
        self._indices_word = dict((i, c) for i, c in enumerate(self._words))

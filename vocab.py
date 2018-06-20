import json

class Vocabulary:
    def __init__(self, chars=None):
        if chars:
            self._chars = ["<UNK>", "<PAD>", "<GO>", "<EOS>"] + chars
            self._char_indices = dict((c, i) for i, c in enumerate(self._chars))
            self._indices_char = dict((i, c) for i, c in enumerate(self._chars))

    def size(self):
        return len(self._chars)

    def char2idx(self, ch):
        if ch not in self._char_indices:
            ch = "<UNK>"
        return self._char_indices[ch]

    def idx2char(self, idx):
        return self._indices_char[idx]

    def save(self, path):
        with open(path, "w") as f:
            f.write(json.dumps(self._chars))

    def load(self, path):
        with open(path, "r") as f:
            s = f.read()
        self._chars = json.loads(s)
        self._char_indices = dict((c, i) for i, c in enumerate(self._chars))
        self._indices_char = dict((i, c) for i, c in enumerate(self._chars))

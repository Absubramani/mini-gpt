from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


class BPETokenizer:
    def __init__(self, vocab_size=1000):
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]")) #BPE tokenizer
        self.tokenizer.pre_tokenizer = Whitespace()

        self.trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[UNK]", "[PAD]"]
        )

    def train(self, files):
        self.tokenizer.train(files, self.trainer)

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

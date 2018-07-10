class Answer:
    """Simple answer wrapper."""
    def __init__(self):
        self.ans2idx = {}
        self.idx2ans = {}
        self.idx = 0
    
    def add_ans(self, ans):
        if not ans in self.ans2idx:
            self.ans2idx[ans] = self.idx
            self.idx2ans[self.idx] = ans
            self.idx += 1
    
    def __call__(self, ans):
        if not ans in self.ans2idx:
            return self.ans2idx["<don't know>"]
        return self.ans2idx[ans]
    
    def __len__(self):
        return len(self.ans2idx)



class Vocabulary:
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
    
    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]
    
    def __len__(self):
        return len(self.word2idx)
class Tokenizer():
    def tokenize(self, data):
        data = "<STRAT> " + data + " <STOP>"
        tokens = data.split()
        return tokens
    
    def count_tokens(self, data, smoothing=False):
        cnt = {'<UNK>': 0}
        words = []        
        for sent in data:
            words.extend(self.tokenize(sent))
        for item in words:
            if item not in cnt.keys():
                cnt.update({item:1})
            else: cnt[item] += 1
        return cnt

    def encode(self, data, smoothing=False):
        i=1
        cnt = self.count_tokens(data, smoothing)
        dict = {'<UNK>':0}
        total = 0
        for k,v in cnt.items():
            if v > 2:
                dict.update({k:i})
                i+=1
            else: cnt['<UNK>'] += v
            total += v
        self.total = total
        self.cnt = cnt
        self.dict = dict
    
    def to_dict(self, sent):
        for word in sent:
            if word not in self.dict:
                word = '<UNK>'                
        return word

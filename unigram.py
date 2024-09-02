import math

from trokenizer import tokenizer

class Unigram():
    def fit(self, data):
        tokenizer = Tokenizer()
        tokenizer.encode(data)
        self.tokenizer = tokenizer
        self.total = tokenizer.total
        self.cnt = tokenizer.cnt
        self.dict = tokenizer.dict

    def probability(self, word, smoothing=False):
        total = self.total
        if smoothing: total += len(self.dict.keys())
        if word in self.dict.keys():
            if smoothing: return self.cnt[word]+1/total
            return self.cnt[word]/total
        else:
            if smoothing: return self.cnt['<UNK>']+1/total 
            return self.cnt['<UNK>']/total

    def perplexity(self, data, smoothing=False):
        words = self.tokenizer.tokenize(data)
        sum = 0
        count = 0
        for word in words:
            prob = self.probability(word, smoothing)
            sum += math.log(prob)
        if math.exp(sum) == 0:            
           return 0
        return 1/((math.exp(sum))**(1/len(words)))
    
    def score(self, data, smoothing=False):
        sum = 0
        for sent in data:
            sum += self.perplexity(sent, smoothing)
            #print(sum)
        return sum/len(data)
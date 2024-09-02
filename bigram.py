import math

from tokenizer import Tokenizer

class Bigram():
    def init(self, data):
        tokenizer = Tokenizer()
        tokenizer.encode(data)
        self.tokenizer = tokenizer
        self.total = tokenizer.total
        self.dict = tokenizer.dict

    def fit(self, data):
        bigrams = []
        cnt = {}
        for sent in data:
            bigrams.extend(self.to_bigram(sent))
        for bi in bigrams:
            if bi not in cnt.keys():
                cnt.update({bi:1})
            else: cnt[bi] += 1
        self.cnt = cnt
        self.prob_dict()

    def to_bigram(self, sent):    
        bigram = []
        for word in sent.split():
            if word not in self.dict.keys():
                sent = sent.replace(word, '<UNK>')
        words = self.tokenizer.tokenize(sent)
        bigram.extend([' '.join(words[i:i+2]) for i in range(len(words)-1)])
        return bigram

    def count_word(self, word1, word2):
        first_sum = 0
        second_sum = 0
        for k, v in self.cnt.items():
            if k.split()[0] == word1:
                first_sum += v
                if k.split()[1] == word2:
                    second_sum = v
        return first_sum, second_sum
    
    def prob_dict(self):
        prob = {}
        i=0
        for keys in self.cnt.keys():
            k1, k2 = keys.split()
            total, count = self.count_word(k1, k2)
            prob.update({keys:(count/total)})
            i += 1
        self.prob = prob
        
    def probability(self, bi, smoothing=False):
        if bi not in self.prob.keys():
            return 1e-10
        return self.prob[bi]
    
    def perplexity(self, data, smoothing=False):
        words = self.to_bigram(data)
        sum = 0
        for word in words:
            prob = self.probability(word, smoothing)
            if prob <= 0: prob = 1e-10
            sum += math.log(prob)
        if math.exp(sum) == 0:            
           return 0
        return 1/((math.exp(sum))**(1/len(words)))
    
    def score(self, data, smoothing=False):
        sum = 0
        for sent in data:
            sum += self.perplexity(sent, smoothing)
        return sum/len(data)
    
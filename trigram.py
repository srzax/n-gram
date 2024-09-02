import math

from tokenizer import tokenizer

class Trigram():
    def __init__(self):
        pass
    
    def init(self, data):
        tokenizer = Tokenizer()
        tokenizer.encode(data)
        self.tokenizer = tokenizer
        self.total = tokenizer.total
        self.dict = tokenizer.dict

    def fit(self, data):
        trigrams = []
        cnt = {}
        for sent in data:
            trigrams.extend(self.to_trigram(sent))
        for tri in trigrams:
            if tri not in cnt.keys():
                cnt.update({tri:1})
            else: cnt[tri] += 1
        self.cnt = cnt
        self.prob_dict()

    def to_trigram(self, sent):    
        trigram = []
        for word in sent.split():
            if word not in self.dict.keys():
                sent = sent.replace(word, '<UNK>')
        words = self.tokenizer.tokenize(sent)
        trigram.extend([' '.join(words[i:i+3]) for i in range(len(words)-2)])
        return trigram

    def count_word(self, word1, word2, word3):
        total = 0
        count = 0
        for k, v in self.cnt.items():
            words = k.split()
            if words[0] == word1 and words[1] == word2:
                total += v
                if words[2] == word3:
                    count = v            
        return total, count
    
    def prob_dict(self):
        prob = {}
        i=0
        for keys in self.cnt.keys():
            k1, k2, k3 = keys.split()
            total, count = self.count_word(k1, k2, k3)
            prob.update({keys:(count/total)})
            i += 1
            print(i)
        self.prob = prob
        
    def probability(self, tri, smoothing=False):
        if tri not in self.prob.keys():
            return 1/(len(self.dict)+len(self.prob))
        return self.prob[tri]
    
    def perplexity(self, data, smoothing=False):
        words = self.to_trigram(data)
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
    
    def save(self):
        with open('tri.pkl', 'wb') as f:
            pickle.dump(self, f)
    
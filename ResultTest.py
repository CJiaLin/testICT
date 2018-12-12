#TF-IDF与ICT结果加权进行排序推荐
import jieba
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

class resulttest():
    def __init__(self,patentinf,word2id,inventor2id,company2id,testinf,topicnum):
        self.testinf = testinf
        self.word2id = word2id
        self.inventor2id = inventor2id
        self.company2id = company2id
        self.patentinf = patentinf
        self.words = []
        self.na = 0.0
        self.topicnum = topicnum
        self.text = []
        self.total_count = {}
        self.word_count = []
        self.stopword = []
        self.total_word = 0
        self.phi = []
        #self.theta = []
        self.tau = []
        self.result = []
        self.similarity = []
        self.topdoc = []

        self.creat_value()

    def creat_value(self):
        with open('./result/'+'tau'+ str(self.topicnum) + '.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            i = 0
            for line in lines:
                self.tau.append([])
                words = line.split('    ')[:-1]
                for word in words:
                    self.tau[i].append(float(word))
                i += 1

        with open('./result/phi'+ str(self.topicnum) + '.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            i = 0
            for line in lines:
                self.phi.append([])
                topics = line.split('    ')[:-1]
                for topic in topics:
                    self.phi[i].append(float(topic))
                i += 1

        with open('./word_result/length.txt', 'r', encoding='utf-8') as length:
            lines = length.readlines()
            total = 0
            for line in lines:
                total += int(line)

            self.na = total / len(lines)

        with open('./word_result/count.txt', 'r', encoding='utf-8') as count:
            lines = count.readlines()
            for line in lines:
                self.total_count[line.split(':')[0]] = int(line.split(':')[1])
                self.total_word += int(line.split(':')[1])

        with open('./word_result/stopword.txt','r', encoding='utf-8') as stopword:
            self.stopword = [line.strip() for line in stopword.readlines()]

        for i in range(len(self.testinf)):
            self.result.append({})
            self.topdoc.append([])
            for j in range(len(self.patentinf)):
                self.result[i][j] = 0.0

        for i in range(len(self.patentinf)):
            self.word_count.append({})


    def topic_probability(self):
        jieba.load_userdict('E:/机器学习/testICT/word_result/userdict_AI.txt')
        docs = []
        i = 0
        for patent in self.testinf:
            self.words.append(jieba.lcut(patent[0]))
            docs.append(' '.join([word for word in self.words[i] if (not word in self.stopword and len(word) != 1)]))
            i += 1

        vectorizer = TfidfVectorizer(lowercase=False)
        tfidf_model = vectorizer.fit_transform(docs)
        words = vectorizer.get_feature_names()
        weight = tfidf_model.toarray()

        result = []

        for item in weight:
            doc = {}
            for i in range(len(words)):
                doc[words[i]] = item[i]

            result.append(sorted(doc.items(),key=lambda item:item[1],reverse=True))

        for i in range(len(result)):
            self.words[i] = [(word[0],word[1]) for word in result[i] if word[1] >= 0.25]

        i = 0
        for patent in tqdm(self.words):
            for j in range(len(self.patentinf)):
                for item in patent:
                    if item[0] in self.patentinf[j][0]:
                        self.result[i][j] += item[1]

                    if item[0] in self.patentinf[j][1]:
                        self.result[i][j] += (item[1] / 3)

                    if item[0] in self.word2id.keys():
                        for k in range(self.topicnum):
                            self.result[i][j] += self.tau[j][k] * self.phi[k][self.word2id[item[0]]]
            i += 1


    def select(self):
        for i in range(len(self.result)):
            self.result[i] = sorted(self.result[i].items(),key=lambda item:item[1],reverse=True)
            self.topdoc[i] = [patent[0] for patent in self.result[i][:10]]

        '''for i in range(len(self.testinf)):
            for j in range(10):
                self.topdoc[i].append(j)

        for i in tqdm(range(len(self.testinf))):
            for j in range(10,len(self.patentinf)):
                for k in range(10):
                    if self.result[i][j] > self.result[i][self.topdoc[i][k]]:
                        self.topdoc[i][k] = j
                        break'''

        return self.topdoc

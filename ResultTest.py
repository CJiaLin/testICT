#TF-IDF与ICT结果加权进行排序推荐
import jieba
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from pandas import Series
import math
import json

class resulttest():
    def __init__(self,patentinf,word2id,inventor2id,company2id,topicnum,testinf = None,sentence=None):
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
        self.theta = []
        self.tau = []
        self.psi = []
        self.result = None
        self.similarity = []
        self.topdoc = []
        self.sen = sentence
        self.tfidf = []

        self.creat_value()

    def creat_value(self):
        with open('./result/tau'+ str(self.topicnum) + '.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            i = 0
            for line in lines:
                self.tau.append([])
                words = line.split('    ')[:-1]
                for word in words:
                    self.tau[i].append(float(word))
                i += 1

        with open('./result/theta'+ str(self.topicnum) + '.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            i = 0
            for line in lines:
                self.theta.append([])
                words = line.split('    ')[:-1]
                for word in words:
                    self.theta[i].append(float(word))
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

        '''with open('./result/psi'+ str(self.topicnum) + '.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            i = 0
            for line in lines:
                self.psi.append([])
                words = line.split('    ')[:-1]
                for word in words:
                    self.psi[i].append(float(word))'''

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

        '''with open('./word_result/tfidf.txt','r',encoding='utf-8') as tfidf:
            lines = tfidf.readlines()
            i = 0
            for line in lines:
                words = line.split('    ')[:-1]
                self.tfidf.append({})
                for word in words:
                    self.tfidf[i][word.split(':')[0]] = float(word.split(':')[1])

                i += 1'''

        for i in range(len(self.patentinf)):
            self.word_count.append({})


    def topic_probability(self):
        jieba.load_userdict('./word_result/userdict_AI.txt')
        for i in range(len(self.testinf)):
            self.result.append({})
            self.topdoc.append([])
            for j in range(len(self.patentinf)):
                self.result[i][j] = 0.0
        docs = []
        i = 0
        for patent in self.testinf:
            self.words.append(jieba.lcut_for_search(patent[0]))
            docs.append(' '.join([word for word in self.words[i] if (not word in self.stopword and len(word) != 1)]))
            i += 1

        vectorizer = TfidfVectorizer(lowercase=False)
        tfidf_model = vectorizer.fit_transform(docs)
        words = vectorizer.get_feature_names()
        weight = tfidf_model.toarray()

        result = []

        for item in weight:
            doc = {}                                #词:TF-IDF
            for i in range(len(words)):
                doc[words[i]] = item[i]

            result.append(sorted(doc.items(),key=lambda item:item[1],reverse=True))

        for i in range(len(result)):
            self.words[i] = [(word[0],word[1]) for word in result[i] if word[1] >= 0.18]

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
        '''for i in range(len(self.result)):
            self.result[i] = sorted(self.result[i].items(),key=lambda item:item[1],reverse=True)
            self.topdoc[i] = [patent[0] for patent in self.result[i][:10]]'''

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


    def Sentence(self):
        jieba.load_userdict('./word_result/userdict_AI.txt')
        words = jieba.lcut_for_search(self.sen)
        words = [word for word in words if (not word in self.stopword and len(word) != 1)]
        middle = []
        self.result = {}

        '''with open('word_result/fenci_final.txt','r', encoding='utf-8') as fencifile:
            lines = fencifile.readlines()
            for line in lines:
                self.text.append(line.strip().split(';')[:-1])

        with open('word_result/patent_count.txt','r', encoding='utf-8') as count:
            lines = count.readlines()
            i = 0
            for line in lines:
                mid = {}
                patent  = line.split('    ')[:-1]
                for word in patent:
                    mid[word.split(':')[0]] = int(word.split(':')[1])

                self.word_count[i] = mid
                i += 1

        for i in range(len(self.patentinf)):
            middle.append(len(self.text[i]) / (len(self.text[i]) + self.na))'''

        for i in range(len(self.patentinf)):
            self.result[self.patentinf[i][4]] = 0.0

        for j in range(len(self.patentinf)):
            for item in words:
                if item in self.patentinf[j][0]:   #title
                    self.result[self.patentinf[j][4]] += self.tfidf[j][item]

                if item in self.patentinf[j][1]:  #abstract
                    self.result[self.patentinf[j][4]] += self.tfidf[j][item]

                if item in self.word2id.keys():
                    for k in range(self.topicnum):
                        for inventor in self.patentinf[j][2]:
                        #self.result[self.patentinf[j][4]] += self.tau[j][k] * self.phi[k][self.word2id[item]]
                            self.result[self.patentinf[j][4]] += (self.theta[self.inventor2id[inventor]][k] * self.phi[k][self.word2id[item]] * (1 / len(self.patentinf[j][2])))

        '''for j in range(len(self.patentinf)):
            for word in words:
                # 在语料库中
                if word in self.word2id.keys():
                    if word in self.word_count[j].keys():
                        plm = ((middle[j] * (self.word_count[j][word] / len(self.text[j]))) + (
                            (1 - middle[j]) * (self.total_count[word] / self.total_word)))
                    else:
                        plm = (((1 - middle[j]) * (self.total_count[word] / self.total_word)))
                    pict = 0.0
                    for k in range(self.topicnum):
                        for inventor in self.patentinf[j][2]:
                            pict += (self.theta[self.inventor2id[inventor]][k] * self.phi[k][self.word2id[word]] * (1 / len(self.patentinf[j][2])))

                    self.result[self.patentinf[j][4]] += plm * pict
                else:
                    pass'''

        self.result = sorted(self.result.items(), key=lambda item:item[1], reverse=True)
        self.topdoc = [patent[0] for patent in self.result[:10]]

        return self.topdoc


    def Patent(self):
        self.result = {}

        if self.testinf == None:
            for i in range(len(self.patentinf)):
                self.result[self.patentinf[i][4]] = {}
                self.topdoc.append([])
                for j in range(len(self.patentinf)):
                    if i != j:
                        self.result[self.patentinf[i][4]][self.patentinf[j][4]] = 0.0
                    else:
                        self.result[self.patentinf[i][4]][self.patentinf[j][4]] = 100.0

            for i in tqdm(range(len(self.patentinf))):
                current = self.tau[i]
                for j in range(len(self.patentinf)):
                    if j != i:
                        for k in range(self.topicnum):
                            self.result[self.patentinf[i][4]][self.patentinf[j][4]] += (math.sqrt(self.tau[j][k]) - math.sqrt(current[k])) ** 2

            for i in range(len(self.result)):
                self.result[self.patentinf[i][4]] = sorted(self.result[self.patentinf[i][4]].items(), key=lambda item:item[1])
                self.topdoc[self.patentinf[i][4]] = [patent[0] for patent in self.result[self.patentinf[i][4]][1:11]]

            with open('./MatchResult/PatentDoc' + str(self.topicnum) + '.json','w',encoding='utf-8') as pt:
                json.dump(self.topdoc,pt,ensure_ascii=False)
        else:
            index = []
            for i in range(len(self.testinf)):
                self.result[self.testinf[i]] = {}
                self.topdoc.append([])
                for j in range(len(self.patentinf)):
                    if self.testinf[i] == int(self.patentinf[j][4]):
                        index.append(j)
                        self.result[self.testinf[i]][self.patentinf[j][4]] = -100
                    else:
                        self.result[self.testinf[i]][self.patentinf[j][4]] = 0.0

            for i in tqdm(range(len(self.testinf))):
                current = self.tau[index[i]]
                for j in range(len(self.patentinf)):
                    if j != i:
                        for k in range(self.topicnum):
                            self.result[self.testinf[i]][self.patentinf[j][4]] += (math.sqrt(self.tau[j][k]) - math.sqrt(current[k])) ** 2

            for i in range(len(self.result)):
                self.result[self.testinf[i][4]] = sorted(self.result[self.testinf[i][4]].items(), key=lambda item:item[1])
                self.topdoc[self.testinf[i][4]] = [patent[0] for patent in self.result[self.testinf[i][4]][1:11]]

            with open('./MatchResult/PatentDoc' + str(self.topicnum) + '.json','w',encoding='utf-8') as pt:
                json.dump(self.topdoc,pt,ensure_ascii=False)

    def Company(self):
        self.result = {}
        self.topdoc = {}
        company2id = {}
        nc = []
        ncsum = []
        with open('./word_result/companymap.txt','r',encoding='utf-8') as cm:
            lines = cm.readlines()
            for line in lines:
                company2id[line.split(':')[0]] = int(line.split(':')[1])

        with open('./middle/nc15-1000.txt','r',encoding='utf-8') as ncfile:
            lines = ncfile.readlines()
            for i in range(len(lines)-1):
                content = lines[i].split('    ')[:-1]
                for j in range(len(content)):
                    if i == 0:
                        nc.append([])
                    nc[j].append(int(content[j]))

        for i in range(len(nc)):
            ncsum.append(0)
            for j in range(len(nc[i])):
                ncsum[i] += nc[i][j]

        for i in range(len(nc)):
            self.psi.append([])
            for j in range(len(nc[i])):
                self.psi[i].append(nc[i][j] / ncsum[i])

        for company1,id1 in company2id.items():
            self.result[company1] = {}
            self.topdoc[company1] = {}
            for company2,id2 in company2id.items():
                self.result[company1][company2] = 0.0

        for company1,id1 in tqdm(company2id.items()):
            current = self.psi[id1]
            for company2,id2 in (company2id.items()):
                if id1 != id2:
                    for k in range(self.topicnum):
                        self.result[company1][company2] += (math.sqrt(self.psi[id2][k]) - math.sqrt(current[k])) ** 2

        for company in company2id:
            self.result[company] = sorted(self.result[company].items(), key=lambda item:item[1])
            self.topdoc[company] = [patent[0] for patent in self.result[company][1:11]]

        with open('./MatchResult/CompanyTop' + str(self.topicnum) + '.json', 'w', encoding='utf-8') as ct:
            json.dump(self.topdoc, ct,ensure_ascii=False)


    def TestSet(self):
        pass
    def count(self):
        with open('word_result/fenci.txt','r', encoding='utf-8') as fencifile:
            lines = fencifile.readlines()
            for line in lines:
                self.text.append(line.strip().split(';')[:-1])

        i = 0
        for content in tqdm(self.text):
            # 转换成DataFrame
            dic = Series(content)
            # 词频统计
            middle = dic.value_counts()
            self.word_count[i] = middle.to_dict()
            i += 1

        with open('word_result/patent_count.txt','w', encoding='utf-8') as count:
            for patent in self.word_count:
                for word,num in patent.items():
                    count.write(word + ':' + str(num) + '    ')
                count.write('\n')



#TF-IDF与ICT结果加权进行排序推荐
import jieba
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from pandas import Series
from collections import defaultdict
from Parm import Parm
import copy
from multiprocessing import Pool
import math
import json

class resulttest():
    def __init__(self,patentinf,word2id,inventor2id,company2id,topicnum,date,testinf = None,sentence=None):
        self.testinf = testinf
        self.word2id = word2id
        self.inventor2id = inventor2id
        self.company2id = company2id
        self.patentinf = patentinf
        self.date = date
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
        self.psi = defaultdict(list)
        self.result = None
        self.similarity = []
        self.topdoc = {}
        self.sen = sentence
        self.tfidf = []
        self.core = 32


        self.creat_value()

    def creat_value(self):
        with open('../SaveOldResult/'+ str(self.topicnum) + '-' + self.date + '/ModelResult/tau.json', 'r', encoding='utf-8') as f:
            self.tau = json.load(f)

        with open('../SaveOldResult/'+ str(self.topicnum) + '-' + self.date + '/ModelResult/theta.json', 'r', encoding='utf-8') as f:
            self.theta = json.load(f)

        with open('../SaveOldResult/'+ str(self.topicnum) + '-' + self.date + '/ModelResult/phi.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            i = 0
            for line in lines:
                self.phi.append([])
                topics = line.split(';')[:-1]
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

        with open('../SaveOldResult/'+ str(self.topicnum) + '-' + self.date + '/WordResult/length.json', 'r', encoding='utf-8') as length:
            leng = json.load(length)
            total = 0
            for k,v in leng.items():
                total += int(v)

            self.na = total / len(leng)

        with open('../SaveOldResult/'+ str(self.topicnum) + '-' + self.date + '/WordResult/count.json', 'r', encoding='utf-8') as count:
            self.total_count = json.load(count)
            for k,v in self.total_count.items():
                self.total_word += v

        with open('../FenciInf/stopword.txt','r', encoding='utf-8') as stopword:
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


    def Sentence(self):
        jieba.load_userdict('./word_result/userdict_AI.txt')
        words = jieba.lcut_for_search(self.sen)
        words = [word for word in words if (not word in self.stopword and len(word) != 1)]
        middle = []
        self.result = {}

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

        self.result = sorted(self.result.items(), key=lambda item:item[1], reverse=True)
        self.topdoc = [patent[0] for patent in self.result[:10]]

        return self.topdoc


    def Patent(self):
        self.result = defaultdict(dict)
        self.topdoc = {}

        for k,v in self.patentinf.items():
            self.result[k] = defaultdict(float)
            self.topdoc[k] = []

        part = int(len(self.patentinf)/self.core)
        result = []
        p = Pool(self.core)
        id = list(self.patentinf.keys())
        for i in range(self.core):
            if i != self.core-1:
                result.append(p.apply_async(func,(self,id[i*part:(i+1)*part],copy.deepcopy(id),copy.deepcopy(self.tau),)))
            else:
                result.append(p.apply_async(func,(self,id[i*part:],copy.deepcopy(id),copy.deepcopy(self.tau),)))

        p.close()
        p.join()

        for item in result:
            res = item.get()
            self.topdoc.update(res)

        with open('./SaveOldResult/' + self.topicnum + '-' + self.date + '/MatchResult/PatentDocID.json','w',encoding='utf-8') as pt:
            json.dump(self.topdoc,pt,ensure_ascii=False,indent=4)

        with open('./SaveOldResult/' + self.topicnum + '-' + self.date + '/MatchResult/PatentDoc.txt','w',encoding='utf-8') as pt:
            for k,v in self.topdoc:
                pt.write(self.patentinf[k]['title'] + ':')
                for item in v:
                    pt.write(self.patentinf[item]['title'] + ';')
                pt.write('\n')

        '''for k,v in tqdm(self.patentinf.items()):
            current = self.tau[k]
            for k1, v1 in self.patentinf.items():
                for i in range(self.topicnum):
                    self.result[k][k1] += (math.sqrt(self.tau[k1][i]) - math.sqrt(current[i])) ** 2

        for k in self.patentinf.keys():
            self.result[k] = sorted(self.result[k].items(), key=lambda item:item[1])
            self.topdoc[k] = [patent[0] for patent in self.result[k][1:11]]

        with open('./SaveOldResult/' + self.topicnum + '-' + self.date + '/MatchResult/PatentDoc.json','w',encoding='utf-8') as pt:
            json.dump(self.topdoc,pt,ensure_ascii=False,indent=4)
        else:
            index = []
            for i in range(len(self.testinf)):
                self.result[self.testinf[i]] = {}
                self.topdoc[self.testinf[i]] = []
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
                json.dump(self.topdoc,pt,ensure_ascii=False)'''

    def Company(self):
        self.result = defaultdict(dict)
        self.topdoc = defaultdict(dict)
        company2id = []
        nc = []
        ncsum = defaultdict(int)
        with open('../SaveOldResult/' + str(self.topicnum) + '-' + self.date + '/WordResult/company.txt','r',encoding='utf-8') as cm:
            lines = cm.readlines()
            for line in lines:
                company2id.append(line.strip())

        with open('../SaveOldResult/' + str(self.topicnum) + '-' + self.date + '/Middle/nc1000.txt','r',encoding='utf-8') as ncfile:
            lines = ncfile.readlines()
            for i in range(len(lines)-1):
                nc.append(json.loads(lines[i]))

        for company in company2id:
            for i in range(len(nc)):
                ncsum[company] += nc[i][company]

        for company in company2id:
            for i in range(len(nc)):
                self.psi[company].append((nc[i][company] + 0.01) / (ncsum[company] + self.topicnum * 0.01))

        for company1 in company2id:
            for company2 in company2id:
                self.result[company1][company2] = 0.0

        for company1 in tqdm(company2id):
            current = self.psi[company1]
            for company2 in (company2id):
                for k in range(self.topicnum):
                    self.result[company1][company2] += (math.sqrt(self.psi[company2][k]) - math.sqrt(current[k])) ** 2

        for company in tqdm(company2id):
            self.result[company] = sorted(self.result[company].items(), key=lambda item:item[1])
            self.topdoc[company] = [patent[0] for patent in self.result[company][1:11]]

        with open('../SaveOldResult/' + str(self.topicnum) + '-' + self.date + '/MatchResult/CompanyTop.json', 'w', encoding='utf-8') as ct:
            json.dump(self.topdoc, ct,ensure_ascii=False)

    def Inventor(self):
        self.result = defaultdict(dict)
        self.topdoc = defaultdict(dict)
        inventor2id = []
        with open('../SaveOldResult/' + str(self.topicnum) + '-' + self.date + '/WordResult/inventor.txt','r',encoding='utf-8') as im:
            lines = im.readlines()
            for line in lines:
                inventor2id.append(line.strip())

        with open('../SaveOldResult/' + str(self.topicnum) + '-' + self.date + '/ModelResult/theta.json','r',encoding='utf-8') as thetafile:
            theta = json.load(thetafile)

        for inventor1 in inventor2id:
            self.result[inventor1] = defaultdict(float)

        part = int(len(inventor2id)/self.core)
        result = []
        p = Pool(self.core)
        for i in range(self.core):
            if i != self.core-1:
                result.append(p.apply_async(func,(self,inventor2id[i*part:(i+1)*part],inventor2id.copy(),theta.copy(),)))
            else:
                result.append(p.apply_async(func,(self,inventor2id[i*part:],inventor2id.copy(),theta.copy(),)))
        '''for inventor1 in tqdm(inventor2id):
            current = theta[inventor1]
            for inventor2 in inventor2id:
                for k in range(self.topicnum):
                    self.result[inventor1][inventor2] += (math.sqrt(self.theta[inventor2][k]) - math.sqrt(current[k])) ** 2'''
        p.close()
        p.join()

        for item in result:
            res = item.get()
            self.topdoc.update(res)

        '''for inventor in tqdm(inventor2id):
            self.result[inventor] = sorted(self.result[inventor].items(), key=lambda item:item[1])
            self.topdoc[inventor] = [patent[0] for patent in self.result[inventor][1:11]]'''

        with open('../SaveOldResult/' + str(self.topicnum) + '-' + self.date + '/MatchResult/InventorTop.json', 'w', encoding='utf-8') as ct:
            json.dump(self.topdoc, ct,ensure_ascii=False,indent=4)

    def par(self,id1,id2,num):
        result = defaultdict(dict)
        topdoc = defaultdict(dict)
        for i1 in id1:
            result[i1] = defaultdict(float)

        for i1 in tqdm(id1):
            current = num[i1]
            for i2 in id2:
                for k in range(self.topicnum):
                    result[i1][i2] += (math.sqrt(num[i2][k]) - math.sqrt(current[k])) ** 2

        for i1 in id1:
            result[i1] = sorted(result[i1].items(), key=lambda item:item[1])
            topdoc[i1] = [patent[0] for patent in result[i1][1:11]]

        return topdoc

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

def func(client, id1,id2,num):
    return client.par(id1,id2,num)


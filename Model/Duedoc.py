# encoding: utf-8
# 利用TF-IDF处理分词
# 利用SubSampling处理分词
import jieba
import re
from Parm import Parm
import json
from collections import defaultdict
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from tqdm import tqdm
import pkuseg
import random
import math
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import logging

logger = logging.getLogger('main.Duedoc')

class Duedoc():
    def __init__(self,docs,topicnum):
        self.docs = docs
        self.content = defaultdict(str)
        self.title = defaultdict(str)
        self.fenci_ori = defaultdict(list)    #去除停用词和高频词的分词结果
        self.fenci_final = defaultdict(list)     #分词结果
        self.length_all = defaultdict(int)        #每篇文档的总词数
        self.length_single = 0       #不重复的词的总词数
        self.total = 0                  #总词数
        self.result = {}
        self.word2id = {}
        self.text = ['']         #所有文档的所有词
        self.edge = 0.10        #高频词界限设定
        self.sample = 0.001

        self.parm = Parm(topicnum)

        i = 0
        for num,doc in self.docs.items():
            self.content[num] = doc['abstract']+doc['sovereignty']
                #re.sub(r'[\n0-9]+',' ',)
            self.title[num] = doc['title']
                #re.sub(r'[\n0-9]+',' ',)
            i += 1

    def due(self):
        #jieba.load_userdict('../FenciInf/' + self.parm.Dict)
        logging.info("分词开始")
        with open('../FenciInf/' + self.parm.StopWordCH, 'r', encoding='utf-8') as stopword:
            stopwords = [line.strip() for line in stopword.readlines()]

        try:
            seg = pkuseg.pkuseg()  # 以默认配置加载模型
            print('分词进度：')
            #text = seg.cut('我爱北京天安门')  # 进行分词
            middle = defaultdict(list)
            for num in tqdm(self.docs.keys()):
                # 分词
                text = ''.join(self.content[num])
                self.content[num] = seg.cut(text)
                self.title[num] = seg.cut(self.title[num])
                #self.content[num] = jieba.lcut(text)
                #self.title[num] = jieba.lcut(self.title[num])

                # 移除停用词
                self.content[num] = [word for word in self.content[num] if (not word in stopwords and len(word) != 1 and not re.match(r'[0-9]+',word))]
                self.title[num] = [word for word in self.title[num] if (not word in stopwords and len(word) != 1 and not re.match(r'[0-9]+',word))]

                middle[num] = {'title':self.title[num],'content':self.content[num]}
                # 保存初始分词结果

            with open(self.parm.ResultFile + '/WordResult/fenci.json', 'w', encoding='utf-8') as fencifile:
                json.dump(middle, fencifile, ensure_ascii=False, indent=4)

            del middle

            logging.info('分词结束！')
        except Exception as e:
            logger.error('分词过程错误！', exc_info=True)

    def duefenciBySubSampling(self):
        logging.info("读取fenci文件")
        text = []
        inventor2id = set()
        company2id = set()
        with open(self.parm.ResultFile + '/WordResult/fenci.json','r', encoding='utf-8') as fencifile:
            fenci = json.load(fencifile)
            for num,doc in fenci.items():
                self.title[num] = doc['title']
                self.content[num] = doc['content']
                text += self.title[num] + self.content[num]
                self.total += len(self.title[num]) + len(self.content[num])

        word_count = FreqDist(text)
        word_count = dict(sorted(word_count.items(), key=lambda  item:item[1],reverse=True))
        with open(self.parm.ResultFile + '/WordResult/count_ori.json', 'w', encoding='utf-8') as countfile:
            json.dump(word_count, countfile, ensure_ascii=False, indent=2)

        text = []
        keys = self.docs.keys()
        for num in keys:
            middle = self.title[num]
            for word in middle:
                ran = (math.sqrt(word_count[word] / (self.sample * self.total)) + 1) * (self.sample * self.total) / word_count[word]
                randm = random.random()
                if ran < randm:
                    self.title[num].remove(word)

            middle = self.content[num]

            for word in middle:
                ran = (math.sqrt(word_count[word] / (self.sample * self.total)) + 1) * (self.sample * self.total) / word_count[word]
                randm = random.random()
                if ran < randm:
                    self.content[num].remove(word)

            text += self.title[num] + self.content[num]

        word_count = FreqDist(text)
        word_count = dict(sorted(word_count.items(), key=lambda  item:item[1],reverse=True))
        self.word2id = {word:i for word,i in zip(word_count.keys(),range(len(word_count)))}
        self.word2id = dict(sorted(self.word2id.items(), key=lambda  item:item[1]))

        with open(self.parm.ResultFile + '/WordResult/count.json', 'w', encoding='utf-8') as countfile:
            json.dump(word_count,countfile,ensure_ascii=False,indent=2)

        with open(self.parm.ResultFile + '/WordResult/length.json', 'w', encoding='utf-8') as lenfile:
            for num in self.title.keys():
                self.length_all[num] = len(self.title[num]) + len(self.content[num])
            json.dump(self.length_all,lenfile,ensure_ascii=False,indent=2)

        with open(self.parm.ResultFile + '/WordResult/wordmap.json','w', encoding='utf-8') as wordmap:
            json.dump(self.word2id,wordmap,ensure_ascii=False,indent=2)

        with open(self.parm.ResultFile + '/WordResult/fenci_final.json', 'w', encoding='utf-8') as fencifile:
            middle = defaultdict(dict)
            for num in self.title.keys():
                middle[num] = {'title': self.title[num], 'content': self.content[num]}

            json.dump(middle,fencifile,ensure_ascii=False,indent=2)

        with open(self.parm.ResultFile + '/WordResult/inventor.txt','w',encoding='utf-8') as inventor:
            with open(self.parm.ResultFile + '/WordResult/company.txt','w',encoding='utf-8') as company:
                for num, doc in self.docs.items():
                    for item in doc['inventor']:
                        inventor2id.add(item)

                    for item in doc['company']:
                        if item not in inventor2id:
                            company2id.add(item)
                        else:
                            company2id.add('0号公司')

                for item in inventor2id:
                    inventor.write(item + '\n')

                for item in company2id:
                    company.write(item + '\n')


class DueEnDoc():
    def __init__(self,docs,topicnum):
        i = 0
        self.docs = docs
        self.content = defaultdict(str)
        self.title = defaultdict(str)
        self.fenci_ori = defaultdict(list)    #去除停用词和高频词的分词结果
        self.fenci_final = defaultdict(list)     #分词结果
        self.length_all = defaultdict(int)        #每篇文档的总词数
        self.length_single = 0       #不重复的词的总词数
        self.total = 0                  #总词数
        self.result = {}
        self.word2id = {}
        self.text = ['']         #所有文档的所有词
        self.edge = 0.10        #高频词界限设定
        self.sample = 0.001

        self.parm = Parm(topicnum)

        i = 0
        for num,doc in self.docs.items():
            self.content[num] = doc['abstract']+doc['sovereignty']
                #re.sub(r'[\n0-9]+',' ',)
            self.title[num] = doc['title']
                #re.sub(r'[\n0-9]+',' ',)
            i += 1

    def due(self):
        logging.info("分词开始")
        #with open('../FenciInf/' + self.parm.StopWordEN, 'r', encoding='utf-8') as stopword:
        #    stopwords = [line.strip() for line in stopword.readlines()]

        sw = stopwords.words('english')

        try:
            print('分词进度：')
            middle = defaultdict(list)
            for num in tqdm(self.docs.keys()):
                # 分词
                text = ''.join(self.content[num])
                self.content[num] = word_tokenize(text)
                self.title[num] = word_tokenize(self.title[num])

                # 移除停用词
                self.content[num] = [word for word in self.content[num] if (not word in sw and len(word) != 1 and not re.match(r'[0-9]+',word))]
                self.title[num] = [word for word in self.title[num] if (not word in sw and len(word) != 1 and not re.match(r'[0-9]+',word))]

                #提取词干
                stemmer =  PorterStemmer()
                self.content[num] = [stemmer.stem(word) for word in self.content[num]]
                self.title[num] = [stemmer.stem(word) for word in self.title[num]]

                middle[num] = {'title':self.title[num],'content':self.content[num]}

                # 保存初始分词结果
                with open(self.parm.ResultFile + '/WordResult/fenci.json', 'w', encoding='utf-8') as fencifile:
                    json.dump(middle, fencifile, ensure_ascii=False, indent=4)
            logging.info('分词结束！')
        except Exception as e:
            logger.error('分词过程错误！', exc_info=True)

    def duefenciByTFIDF(self):
        logging.info("处理初步分词结果")
        self.seg_lists = []
        self.seg_list = []
        logging.info("读取fenci.txt文件")
        with open('word_result/fenci.txt','r', encoding='utf-8') as fencifile:
            lines = fencifile.readlines()
            i = 0
            for line in lines:
                self.title[i] = line.strip('\n').split('----')[0].split(';')[:-1]
                self.content[i] = line.strip('\n').split('----')[1].split(';')[:-1]
                text = ' '.join(line.strip('\n').split('----')[0].split(';')[:-1]) + ' ' + ' '.join(line.strip('\n').split('----')[1].split(';')[:-1])
                self.seg_list.append(text)
                i += 1

        vectorizer = CountVectorizer(lowercase=False)
        transformer = TfidfTransformer()
        patent_count = vectorizer.fit_transform(self.seg_list)
        patent_count = patent_count.toarray()
        count_ori = np.sum(patent_count,axis=0)
        vector = vectorizer.get_feature_names()      #所有分词
        idf_result = transformer.fit_transform(patent_count)
        tfidf = idf_result.toarray()   # 对应的tf-idf值
        self.word2id = vectorizer.vocabulary_

        word_count_ori = dict(zip(vector,count_ori))
        word_count_ori = dict(sorted(word_count_ori.items(),key=lambda  item:item[1],reverse=True))
        #保存词频统计结果
        with open('word_result/count_ori.txt', 'w', encoding='utf-8') as countfile:
            for word,num in word_count_ori.items():
                countfile.write(word + ':' + str(num) + '\n')

        for m in range(len(self.title)):
            n = 0
            i = 0
            length = len(self.title[m])
            while n < length:
                word = self.title[m][i]
                if word not in self.word2id.keys():
                    self.title[m].remove(word)
                    i -= 1
                else:
                    id = self.word2id[word]
                    if tfidf[m][id] < self.edge:
                        self.title[m].remove(word)
                        i -= 1
                n += 1
                i += 1

            n = 0
            i = 0
            length = len(self.content[m])
            while n < length:
                word = self.content[m][i]
                if word not in self.word2id.keys():
                    self.content[m].remove(word)
                    i -= 1
                else:
                    id = self.word2id[word]
                    if tfidf[m][id] < self.edge:
                        self.content[m].remove(word)
                        i -= 1
                n += 1
                i += 1

        self.seg_list = []
        for i in range(len(self.title)):
            self.seg_list.append(' '.join(self.title[i]) + ' ' + ' '.join(self.content[i]))

        patent_count = vectorizer.fit_transform(self.seg_list)
        patent_count = patent_count.toarray()
        vector = vectorizer.get_feature_names()      #所有分词
        count = np.sum(patent_count,axis=0)
        word_count = dict(zip(vector,count))
        word_count = dict(sorted(word_count.items(),key=lambda  item:item[1],reverse=True))
        #保存词频统计结果
        with open('word_result/count.txt', 'w', encoding='utf-8') as countfile:
            for word,num in word_count.items():
                countfile.write(word + ':' + str(num) + '\n')

        with open('word_result/wordmap.txt','w', encoding='utf-8') as wordmap:
            for key,value in self.word2id.items():
                wordmap.write(key + ':' + str(value) + '\n')

        id2word = {value:key for key,value in self.word2id.items()}
        with open('word_result/tfidf.txt', 'w', encoding='utf-8') as tfidffile:
            print('tfidf保存进度：')
            for patent in tqdm(tfidf):
                i = 0
                for value in patent:
                    if value != 0.0:
                        tfidffile.write(id2word[i] + ':' + str(value) + '    ')
                    i += 1

                tfidffile.write('\n')

        with open('word_result/fenci_final.txt', 'w', encoding='utf-8') as fencifile:
            for i in range(len(self.title)):
                for word in self.title[i]:
                    fencifile.write(word + ';')

                fencifile.write('----')

                for word in self.content[i]:
                    fencifile.write(word + ';')
                fencifile.write('\n')

        with open('word_result/length.txt', 'w', encoding='utf-8') as lenfile:
            for i in range(len(self.seg_list)):
                self.length_all[i] = len(self.title[i]) + len(self.content[i])
                lenfile.write(str(self.length_all[i]) + '\n')
        
        logging.info("分词结果、词频统计结果保存")
        #return self.docs

    def duefenciBySubSampling(self):
        logging.info("读取fenci文件")
        text = []
        inventor2id = set()
        company2id = set()
        with open(self.parm.ResultFile + '/WordResult/fenci.json', 'r', encoding='utf-8') as fencifile:
            fenci = json.load(fencifile)
            for num, doc in fenci.items():
                self.title[num] = doc['title']
                self.content[num] = doc['content']
                text += self.title[num] + self.content[num]
                self.total += len(self.title[num]) + len(self.content[num])

        word_count = FreqDist(text)
        word_count = dict(sorted(word_count.items(), key=lambda item: item[1], reverse=True))
        with open(self.parm.ResultFile + '/WordResult/count_ori.json', 'w', encoding='utf-8') as countfile:
            json.dump(word_count, countfile, ensure_ascii=False, indent=2)

        text = []
        keys = self.docs.keys()
        for num in keys:
            middle = self.title[num]
            for word in middle:
                ran = (math.sqrt(word_count[word] / (self.sample * self.total)) + 1) * (self.sample * self.total) / \
                      word_count[word]
                randm = random.random()
                if ran < randm:
                    self.title[num].remove(word)

            middle = self.content[num]

            for word in middle:
                ran = (math.sqrt(word_count[word] / (self.sample * self.total)) + 1) * (self.sample * self.total) / \
                      word_count[word]
                randm = random.random()
                if ran < randm:
                    self.content[num].remove(word)

            text += self.title[num] + self.content[num]

        word_count = FreqDist(text)
        word_count = dict(sorted(word_count.items(), key=lambda item: item[1], reverse=True))
        self.word2id = {word: i for word, i in zip(word_count.keys(), range(len(word_count)))}
        self.word2id = dict(sorted(self.word2id.items(), key=lambda item: item[1]))

        with open(self.parm.ResultFile + '/WordResult/count.json', 'w', encoding='utf-8') as countfile:
            json.dump(word_count, countfile, ensure_ascii=False, indent=2)

        with open(self.parm.ResultFile + '/WordResult/length.json', 'w', encoding='utf-8') as lenfile:
            for num in self.title.keys():
                self.length_all[num] = len(self.title[num]) + len(self.content[num])
            json.dump(self.length_all, lenfile, ensure_ascii=False, indent=2)

        with open(self.parm.ResultFile + '/WordResult/wordmap.json', 'w', encoding='utf-8') as wordmap:
            json.dump(self.word2id, wordmap, ensure_ascii=False, indent=2)

        with open(self.parm.ResultFile + '/WordResult/fenci_final.json', 'w', encoding='utf-8') as fencifile:
            middle = defaultdict(dict)
            for num in self.title.keys():
                middle[num] = {'title': self.title[num], 'content': self.content[num]}

            json.dump(middle, fencifile, ensure_ascii=False, indent=2)

        with open(self.parm.ResultFile + '/WordResult/inventor.txt', 'w', encoding='utf-8') as inventor:
            with open(self.parm.ResultFile + '/WordResult/company.txt', 'w', encoding='utf-8') as company:
                for num, doc in self.docs.items():
                    for item in doc['inventor']:
                        inventor2id.add(item)

                    for item in doc['company']:
                        if item not in inventor2id and item != ['nan']:
                            company2id.add(item)
                        else:
                            company2id.add('0号公司')

                for item in inventor2id:
                    inventor.write(item + '\n')

                for item in company2id:
                    company.write(item + '\n')

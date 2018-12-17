# encoding: utf-8
# 计算每篇文档词的TF-IDF
import jieba
import re
#from nltk.stem import PorterStemmer
#from nltk.tokenize import word_tokenize
#from nltk.corpus import stopwords
from nltk.probability import FreqDist
from tqdm import tqdm
import random
#import pandas as pd
import math
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import logging


logging.basicConfig(level=logging.DEBUG,
                    filename='logging.log',
                    filemode='a',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s-%(message)s')
logger = logging.getLogger(__name__)

class Duedoc():
    def __init__(self,docs):
        i = 0
        self.docs = docs
        self.content = []
        self.title = []
        self.seg_lists = []    #去除停用词和高频词的分词结果
        self.seg_list = []     #分词结果
        self.length_all = []        #每篇文档的总词数
        self.length_single = 0       #不重复的词的总词数
        self.total = 0                  #总词数
        #self.dic = pd.DataFrame()
        self.result = {}
        self.word2id = {}
        self.text = ['']         #所有文档的所有词
        #self.hotword = []      #记录高频词汇
        self.edge = 0.10        #高频词界限设定
        self.sample = 0.001

        while i < len(docs):
            self.content.append([])
            self.title.append([])
            self.seg_lists.append([])
            self.seg_list.append([])
            self.length_all.append([0])
            i += 1

        i = 0
        for doc in self.docs:
            self.content[i] = re.sub(r'[\n0-9]*','',doc[1])
            self.title[i] = re.sub(r'[0-9]','',doc[0])
            i += 1

    def due(self):
        jieba.load_userdict('./word_result/userdict_AI.txt')
        logging.info("分词开始")
        with open('word_result/fenci.txt', 'w', encoding='utf-8') as resultfile:
            resultfile.write('')

        with open('word_result/stopword.txt', encoding='utf-8') as stopword:
            stopwords = [line.strip() for line in stopword.readlines()]

        try:
            print('分词进度：')
            for i in tqdm(range(len(self.content))):
                # 分词
                text = ''.join(self.content[i])
                self.seg_lists[i] = jieba.lcut(text)
                self.title[i] = jieba.lcut(self.title[i])

                # 移除停用词
                self.seg_list[i] = [word for word in self.seg_lists[i] if (not word in stopwords and len(word) != 1)]
                self.title[i] = [word for word in self.title[i] if (not word in stopwords and len(word) != 1)]

                # 保存初始分词结果
                with open('word_result/fenci.txt', 'a', encoding='utf-8') as fencifile:
                    for word in self.title[i]:
                        fencifile.write(word + ';')

                    fencifile.write('----')

                    for word in self.seg_list[i]:
                        fencifile.write(word + ';')
                    fencifile.write('\n')
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
                    if tfidf[m][id] < self.edge and word_count_ori[word] > 2000:
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
        logging.info("读取fenci.txt文件")
        text = []
        with open('word_result/fenci.txt','r', encoding='utf-8') as fencifile:
            lines = fencifile.readlines()
            i = 0
            for line in lines:
                self.title[i] = line.strip('\n').split('----')[0].split(';')[:-1]
                self.content[i] = line.strip('\n').split('----')[1].split(';')[:-1]
                text += self.title[i] + self.content[i]
                self.total += len(self.title[i]) + len(self.content[i])
                i += 1

        word_count = FreqDist(text)
        word_count = dict(sorted(word_count.items(), key=lambda  item:item[1],reverse=True))
        with open('word_result/count_ori.txt', 'w', encoding='utf-8') as countfile:
            for word,num in word_count.items():
                countfile.write(word + ':' + str(num) + '\n')

        text = []
        for i in range(len(self.title)):
            n = 0
            for j in range(len(self.title[i])):
                word = self.title[i][n]
                ran = (math.sqrt(word_count[word] / (self.sample * self.total)) + 1) * (self.sample * self.total) / word_count[word]
                randm = random.random()
                if ran < randm:
                    self.title[i].remove(word)
                    n -= 1

                n += 1

            n = 0
            for j in range(len(self.content[i])):
                word = self.content[i][n]
                ran = (math.sqrt(word_count[word] / (self.sample * self.total)) + 1) * (self.sample * self.total) / word_count[word]
                randm = random.random()
                if ran < randm:
                    self.content[i].remove(word)
                    n -= 1
                n += 1

            text += self.title[i] + self.content[i]

        word_count = FreqDist(text)
        word_count = dict(sorted(word_count.items(), key=lambda  item:item[1],reverse=True))
        self.word2id = {word:i for word,i in zip(word_count.keys(),range(len(word_count)))}
        self.word2id = dict(sorted(self.word2id.items(), key=lambda  item:item[1]))
        with open('word_result/count.txt', 'w', encoding='utf-8') as countfile:
            for word,num in word_count.items():
                countfile.write(word + ':' + str(num) + '\n')

        with open('word_result/length.txt', 'w', encoding='utf-8') as lenfile:
            for i in range(len(self.title)):
                self.length_all[i] = len(self.title[i]) + len(self.content[i])
                lenfile.write(str(self.length_all[i]) + '\n')

        with open('word_result/wordmap.txt','w', encoding='utf-8') as wordmap:
            for key,value in self.word2id.items():
                wordmap.write(key + ':' + str(value) + '\n')

        with open('word_result/fenci_final.txt', 'w', encoding='utf-8') as fencifile:
            for i in range(len(self.title)):
                for word in self.title[i]:
                    fencifile.write(word + ';')

                fencifile.write('----')

                for word in self.content[i]:
                    fencifile.write(word + ';')
                fencifile.write('\n')

'''class DueEnDoc():
    def __init__(self,docs):
        i = 0
        self.docs = docs
        self.content = []
        self.title = []
        self.seg_lists = []    #去除停用词和高频词的分词结果
        self.seg_list = []     #分词结果
        self.length_all = []        #每篇文档的总词数
        self.length_single = 0       #不重复的词的总词数
        self.total = 0              #总词数
        self.sample = 0.001
        #self.dic = pd.DataFrame()
        self.result = {}
        self.word2id = {}
        self.text = ['']         #所有文档的所有词
        #self.hotword = []      #记录高频词汇
        self.edge = 1        #高频词界限设定

        while i < len(docs):
            self.content.append([])
            self.title.append([])
            self.seg_lists.append([])
            self.seg_list.append([])
            self.length_all.append([0])
            i += 1

        i = 0
        for doc in self.docs:
            self.content[i] = re.sub(r'[\n0-9]*','',doc[1])
            self.title[i] = re.sub(r'[0-9]','',doc[0])
            i += 1

    def due(self):
        logging.info("分词开始")
        with open('word_result/fenci.txt', 'w', encoding='utf-8') as resultfile:
            resultfile.write('')

        sw = stopwords.words('english')

        try:
            print('分词进度：')
            for i in tqdm(range(len(self.content))):
                # 分词
                text = ''.join(self.content[i])
                self.seg_lists[i] = word_tokenize(text)
                self.title[i] = word_tokenize(self.title[i])

                # 移除停用词
                self.seg_list[i] = [word for word in self.seg_lists[i] if (not word in sw and len(word) != 1)]
                self.title[i] = [word for word in self.title[i] if (not word in sw and len(word) != 1)]

                #提取词干
                stemmer =  PorterStemmer()
                self.seg_lists[i] = [stemmer.stem(word) for word in self.seg_list[i]]

                # 保存初始分词结果
                with open('word_result/fenci.txt', 'a', encoding='utf-8') as fencifile:
                    for word in self.title[i]:
                        fencifile.write(word + ';')

                    fencifile.write('----')

                    for word in self.seg_lists[i]:
                        fencifile.write(word + ';')
                    fencifile.write('\n')
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
        logging.info("读取fenci.txt文件")
        self.seg_list = []
        text = []
        with open('word_result/fenci.txt','r', encoding='utf-8') as fencifile:
            lines = fencifile.readlines()
            i = 0
            for line in lines:
                self.title[i] = line.strip('\n').split('----')[0].split(';')[:-1]
                self.content[i] = line.strip('\n').split('----')[1].split(';')[:-1]
                text += self.title[i] + self.content[i]
                self.total += len(self.title[i]) + len(self.content[i])
                i += 1

        word_count = FreqDist(text)
        word_count = dict(sorted(word_count.items(), key=lambda  item:item[1],reverse=True))
        with open('word_result/count_ori.txt', 'w', encoding='utf-8') as countfile:
            for word,num in word_count.items():
                countfile.write(word + ':' + str(num) + '\n')

        text = []
        for i in range(len(self.title)):
            n = 0
            for j in range(len(self.title[i])):
                word = self.title[i][n]
                ran = (math.sqrt(word_count[word] / (self.sample * self.total)) + 1) * (self.sample * self.total) / word_count[word]
                randm = random.random()
                if ran < randm:
                    self.title[i].remove(word)
                    n -= 1

                n += 1

            n = 0
            for j in range(len(self.content[i])):
                word = self.content[i][n]
                ran = (math.sqrt(word_count[word] / (self.sample * self.total)) + 1) * (self.sample * self.total) / word_count[word]
                randm = random.random()
                if ran < randm:
                    self.content[i].remove(word)
                    n -= 1
                n += 1

            text += self.title[i] + self.content[i]

        word_count = FreqDist(text)
        word_count = dict(sorted(word_count.items(), key=lambda  item:item[1],reverse=True))
        with open('word_result/count.txt', 'w', encoding='utf-8') as countfile:
            for word,num in word_count.items():
                countfile.write(word + ':' + str(num) + '\n')
                
        with open('word_result/length.txt', 'w', encoding='utf-8') as lenfile:
            for i in range(len(self.seg_list)):
                self.length_all[i] = len(self.title[i]) + len(self.content[i])
                lenfile.write(str(self.length_all[i]) + '\n')
'''
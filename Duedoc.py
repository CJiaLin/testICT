# encoding: utf-8
import jieba
import re
from tqdm import tqdm
import pandas as pd
from pandas import Series
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
        self.length_all = []        #总词数
        self.length_single = 0       #不重复的词的总词数
        self.dic = pd.DataFrame()
        self.result = None
        self.word2id = {}
        self.text = []         #所有文档的所有词
        self.hotword = []      #记录高频词汇
        self.edge = 3000000        #高频词界限设定

        while i < len(docs):
            self.content.append([])
            self.title.append([])
            self.seg_lists.append([])
            self.seg_list.append([])
            self.length_all.append([0])

            i += 1
        i = 0
        for doc in self.docs:
            self.title[i] = re.sub(r'[\n0-9]*','',doc[0])
            self.content[i] = re.sub(r'[\n0-9]*','',doc[1])
            i += 1

    def due(self):
        logging.info("分词开始")
        jieba.load_userdict('E:/机器学习/testICT/word_result/userdict_AI.txt')
        with open('word_result/fenci.txt', 'w', encoding='utf-8') as resultfile:
            resultfile.write('')

        with open('word_result/stopword.txt', encoding='utf-8') as st:
            sw = [line.strip() for line in st.readlines()]

        try:
            for i in tqdm(range(len(self.content))):
                # 分词
                text = ''.join(self.content[i])
                self.seg_lists[i] = jieba.lcut(text)
                self.title[i] = jieba.lcut(self.title[i])

                # 移除停用词
                self.seg_list[i] = [word for word in self.seg_lists[i] if (not word in sw and len(word) != 1)]
                self.title[i] = [word for word in self.title[i] if (not word in sw and len(word) != 1)]

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


    def duefenci(self):
        logging.info("处理初步分词结果")
        with open('word_result/fenci_final.txt', 'w', encoding='utf-8') as finalfile:
            finalfile.write('')

        with open('word_result/count.txt','w', encoding='utf-8') as countfile:
            countfile.write('')

        with open('word_result/length.txt', 'w', encoding='utf-8') as lenfile:
            lenfile.write('')

        title = []
        self.seg_lists = []
        self.seg_list = []
        logging.info("读取fenci.txt文件")
        with open('word_result/fenci.txt','r', encoding='utf-8') as fencifile:
            lines = fencifile.readlines()
            for line in lines:
                self.text.extend(line.strip('\n').split('----')[0].split(';')[:-1] + line.strip('\n').split('----')[1].split(';')[:-1])
                title.append(line.strip('\n').split('----')[0].split(';')[:-1])
                self.seg_list.append(line.strip('\n').split('----')[1].split(';')[:-1])

        # 转换成DataFrame
        self.dic = Series(self.text)
        #print(self.dic)
        logging.info("DataFrame数据格式转换成功！")
        # 词频统计
        self.result = self.dic.value_counts()
        #print(self.result)
        logging.info("词频统计结束！")
        # 统计结果转换成dict
        self.result = self.result.to_dict()
        # 统计高频词汇
        for key,value in self.result.items():
            if int(value) >= self.edge:
                self.hotword.append(key)
        # 移除高频词汇
        for i in range(len(self.seg_list)):
            self.seg_lists.append([])
            self.seg_lists[i] = [word for word in self.seg_list[i] if not word in self.hotword]
            self.title[i] = [word for word in title[i] if not word in self.hotword]
        logging.info("高频词移除")
        # 移除高频词的词频统计结果
        for item in range(len(self.hotword)):
            self.result.pop(self.hotword[item])

        self.length_single = len(self.result)
        # 总词数
        with open('word_result/length.txt', 'a', encoding='utf-8') as lenfile:
            for i in range(len(self.seg_lists)):
                self.length_all[i] = len(self.seg_lists[i]) + len(self.title[i])
                lenfile.write(str(self.length_all[i]) + '\n')
        #self.text.extend(self.seg_lists[i])

        #保存分词结果
        for i in range(len(self.docs)):
            self.docs[i][0] = self.title[i]
            self.docs[i][1] = self.seg_lists[i]

        with open('word_result/fenci_final.txt','a', encoding='utf-8') as fencifile:
            for i in range(len(self.title)):
                for word in self.title[i]:
                    fencifile.write(word + ';')

                fencifile.write('----')

                for word in self.seg_lists[i]:
                    fencifile.write(word + ';')
                fencifile.write('\n')

        #保存词频统计结果
        with open('word_result/count.txt', 'a', encoding='utf-8') as countfile:
            for key,value in self.result.items():
                countfile.write(key + ':' + str(value) + '\n')
        logging.info("分词结果、词频统计结果保存")
        #return self.docs

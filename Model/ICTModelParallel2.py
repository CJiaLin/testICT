#并行化ICT模型算法
#import numpy as np
import random
import json
from tqdm import tqdm
import logging
from collections import defaultdict
from multiprocessing import Pool
import copy

logger = logging.getLogger('main.ICT')

class ICT():
    word = []     #存储第m篇文档的第n个词被指定的Topic id   M*per_doc_word_len
    inventor = []   #存储第m篇文档的第n个词指定的发明者ID     M*per_doc_word_len
    nd = []       #第i篇文档中第j个Topic产生的词的个数   M*K
    ndsum = []    #第i篇文档的总词数   M
    nw = []       #第i个Topic中第j个词的个数   K*V
    nwsum = []    #第i个Topic的总词数    K
    m = {}        #第i个发明人下Topic j出现次数  A*K
    msum = defaultdict(int)     #第i个发明人的所有主题出现次数和 A
    nc = []       #主题i中公司j出现的次数     K*C
    ncsum = []    #主题i中所有公司出现的次数总和  C

    patent = []    #将文档切割成几部分进行并行训练

    theta = {}    #inventor-topic 二维矩阵，存储每个inventor下Topic的概率 A*K
    phi = []      #topic-word 二维矩阵，存储每个Topic下Word的概率    K*V
    psi = []      #company-topic 二维矩阵，存储每个company下Topic的概率   K*C
    tau = {}      #patent-topic 二维矩阵，存储每个Patent下Topic的概率
    tassign = defaultdict(list)

    def __init__(self,topicnum,iternum,length_all,length_single,fenciinf,word2id,inventor2id,company2id,parm,middle=0,newdata=None):
        self.parm = parm
        self.K = int(topicnum)      #Topic数
        self.A = len(inventor2id)   #发明者人数
        self.C = len(company2id)    #公司数
        self.iternum = int(iternum) #迭代次数
        self.alpha = 50.0/self.K
        self.beta = 0.01
        self.mu = 0.01
        self.omega = 0.01
        self.doclength = length_all     #每篇文档的总词数
        self.V = length_single      #不重复词的个数
        self.word2id = word2id
        self.inventor2id = inventor2id
        self.company2id = company2id
        self.fenciinf = []           #文档分词结果
        self.savenum = self.parm.savenum  #每迭代多少次保存参数
        self.program = self.parm.core
        self.Kalpha = self.K * self.alpha
        self.Vbeta = self.V * self.beta
        self.Cmu = self.C * self.mu
        self.middle = middle
        self.id2word = {v:k for k,v in self.word2id.items()}

        for num,doc in fenciinf.items():
            self.fenciinf.append([doc['fenci'],doc['inventor'],doc['company'],num])

        #self.fenciinf = self.fenciinf
        self.part = int(len(self.fenciinf)/self.program)        #每一部分训练集大小

        self.M = len(self.fenciinf)     #文档数

        try:
            self.creat_value()
            logging.info("创建数据成功！")
            self.init_value()
            logging.info("初始化数据成功！")
        except Exception as e:
            logging.error("数据初始化失败！", exc_info=True)

        if middle == 0:
            self.iter()

    #创建数据
    def creat_value(self):
        for doc in self.fenciinf:
            num = doc[3]
            self.tau[num] = [0] * self.K
            self.word.append([])
            self.inventor.append([])
            self.nd.append([0] * self.K)
            self.ndsum.append(0)
            for i in range(len(doc[2])):
                if doc[2][i] in self.inventor2id:
                    doc[2][i] = '0号公司'

        for inv_id in self.inventor2id:
            self.theta[inv_id] = [0] * self.K
            self.m[inv_id] = [0] * self.K

        for topic_id in range(self.K):
            self.phi.append([0] * len(self.word2id))
            self.psi.append(defaultdict(int))
            self.nw.append([0] * len(self.word2id))
            self.nwsum.append(0)
            self.nc.append(defaultdict(int))
            self.ncsum.append(0)

    #初始化数据
    def init_value(self):
        i = 0
        print('初始化数据：')
        for doc in tqdm(self.fenciinf):
            for word in doc[0]:
                topic_id = random.randint(0,self.K-1)     #选取主题
                ran_inv = random.randint(0,len(doc[1])-1)
                inventor_id = doc[1][ran_inv]
                word_id = self.word2id[word]

                self.word[i].append(topic_id)
                self.inventor[i].append(inventor_id)

                self.nw[topic_id][word_id] += 1
                self.nwsum[topic_id] += 1
                self.nd[i][topic_id] += 1
                self.ndsum[i] += 1

                self.m[inventor_id][topic_id] += 1
                self.msum[inventor_id] += 1

                for company in doc[2]:
                    self.nc[topic_id][company] += 1
                    self.ncsum[topic_id] += 1

            i += 1

    #迭代word-topic分布情况，同时计算theta,phi,psi
    def iter(self):
        logging.info("开始迭代")
        if self.middle == 0:
            self.middle_save(0)

        for t in tqdm(range(int(self.iternum/self.savenum))):
            nw = [None] * self.program
            nwsum = [None] * self.program
            m = [None] * self.program
            msum = [None] * self.program
            nc = [None] * self.program
            ncsum = [None] * self.program
            nd = [None] * self.program
            word = [None] * self.program
            inventor = [None] * self.program

            result = []
            p = Pool(self.program)
            try:
                for j in range(self.program):
                    if j != self.program-1:
                        result.append(p.apply_async(func, (self, self.fenciinf[j*self.part:(j+1)*self.part],
                                                           self.word[j*self.part:(j+1)*self.part],
                                                           self.inventor[j*self.part:(j+1)*self.part],
                                                           self.nd[j*self.part:(j+1)*self.part],
                                                           copy.deepcopy(self.nw), copy.deepcopy(self.nwsum),
                                                           copy.deepcopy(self.m.copy()),copy.deepcopy(self.msum),
                                                           copy.deepcopy(self.nc), copy.deepcopy(self.ncsum),)))
                    else:
                        result.append(p.apply_async(func, (self, self.fenciinf[j*self.part:],
                                                           self.word[j*self.part:],
                                                           self.inventor[j*self.part:],
                                                           self.nd[j*self.part:],
                                                           copy.deepcopy(self.nw), copy.deepcopy(self.nwsum),
                                                           copy.deepcopy(self.m),copy.deepcopy(self.msum),
                                                           copy.deepcopy(self.nc), copy.deepcopy(self.ncsum),)))
            except Exception as e:
                logging.error("第"+str((t+1) * int(self.iternum/10))+"次迭代失败", exc_info = True)

            p.close()
            p.join()

            del p

            i = 0
            for res in result:
                word[i],inventor[i],nd[i],nw[i],nwsum[i],m[i],msum[i],nc[i],ncsum[i] = res.get()
                i += 1

            for topic_id in range(self.K):
                self.nwsum[topic_id] = 0
                for word_id in range(self.V):
                    new = 0
                    for k in range(len(nw)):
                        new += nw[k][topic_id][word_id] - self.nw[topic_id][word_id]
                    self.nw[topic_id][word_id] += new
                    self.nwsum[topic_id] += self.nw[topic_id][word_id]

            for inv_id in self.inventor2id:
                self.msum[inv_id] = 0
                for topic_id in range(self.K):
                    new = 0
                    for k in range(len(m)):
                        new += m[k][inv_id][topic_id] - self.m[inv_id][topic_id]
                    self.m[inv_id][topic_id] += new
                    self.msum[inv_id] += self.m[inv_id][topic_id]

            for topic_id in range(self.K):
                self.ncsum[topic_id] = 0
                for com_id in self.company2id:
                    new = 0
                    for k in range(len(nc)):
                        new += nc[k][topic_id][com_id] - self.nc[topic_id][com_id]
                    self.nc[topic_id][com_id] += new
                    self.ncsum[topic_id] += self.nc[topic_id][com_id]

            self.word = []
            for item in word:
                self.word += item

            self.inventor = []
            for item in inventor:
                self.inventor += item

            self.nd = []
            for item in nd:
                self.nd += item

            self.middle_save(((t+1) * self.savenum) + self.middle)

        for inv_id in self.inventor2id:
            for k in range(self.K):
                self.theta[inv_id][k] = (self.m[inv_id][k] + self.alpha) / (self.msum[inv_id] + self.K * self.alpha)
        logging.info("theta计算完成")
        for topic_id in range(self.K):
            for word_id in range(self.V):
                self.phi[topic_id][word_id] = (self.nw[topic_id][word_id] + self.beta) / (self.nwsum[topic_id] + self.V * self.beta)
        logging.info("phi计算完成")
        for com_id in self.company2id:
            for topic_id in range(self.K):
                self.psi[topic_id][com_id] = (self.nc[topic_id][com_id] + self.mu) / (self.ncsum[topic_id] + self.C * self.mu)
        logging.info("psi计算完成")
        for m in range(self.M):
            for k in range(self.K):
                self.tau[self.fenciinf[m][3]][k] = (self.nd[m][k] + self.omega) / (self.ndsum[m] + self.K * self.omega)
        logging.info('tau计算完成')

    #主进程，吉布斯采样过程
    def process(self, patentinf, word, inventor, nd, nw, nwsum, m, msum, nc, ncsum):
        for i in tqdm(range(self.savenum)):
            j = 0
            for p in patentinf:
                num = p[3]
                for q in range(self.doclength[num]):
                    topic_id = word[j][q]
                    inventor_id = inventor[j][q]
                    word_id = self.word2id[p[0][q]]

                    nw[topic_id][word_id] -= 1
                    nwsum[topic_id] -= 1

                    m[inventor_id][topic_id] -= 1
                    msum[inventor_id] -= 1

                    for company in p[2]:
                        nc[topic_id][company] -= 1
                    ncsum[topic_id] -= 1

                    nd[j][topic_id] -= 1

                    newtopic, newinventor = self.probability(patentinf,nc,ncsum,m,msum,nw,nwsum,p,q)

                    word[j][q] = newtopic
                    inventor[j][q] = newinventor

                    nw[newtopic][word_id] += 1
                    nwsum[newtopic] += 1

                    m[newinventor][newtopic] += 1
                    msum[newinventor] += 1

                    for company in p[2]:
                        nc[newtopic][company] += 1
                    ncsum[newtopic] += 1

                    nd[j][newtopic] += 1

                j += 1

        return word,inventor,nd,nw,nwsum,m,msum,nc,ncsum

    # 计算每个Topic的概率
    def probability(self, patentinf, nc, ncsum, m, msum, nw, nwsum, p, q):
        word_id = self.word2id[p[0][q]]
        topic_id = 0
        gailv = []
        test = []
        i = 0
        for inventor in p[1]:
            inventor_id = inventor
            test.append([])
            for k in range(self.K):
                company_p = 0.0
                for company in p[2]:
                    company_p += (nc[k][company] + self.mu) / (ncsum[k] + self.Cmu)

                result = ((m[inventor_id][k] + self.alpha) / (msum[inventor_id] + self.Kalpha)) * (((nw[k][word_id] + self.beta) / (nwsum[k] + self.Vbeta)) + (company_p / len(p[2])))
                test[i].append(result)
                gailv.append(result)
            i += 1

        for j in range(1, len(gailv)):
            gailv[j] += gailv[j - 1]

        u = random.random() * gailv[-1]

        for i in range(len(gailv)):
            if gailv[i] > u:
                topic_id = i
                break

        inventor = int(topic_id / self.K)
        newtopic = topic_id - inventor * self.K
        newinventor = p[1][inventor]

        return newtopic, newinventor

    #保存结果
    def save(self):
        with open(self.parm.ResultFile + '/ModelResult/theta.json','w',encoding='utf-8') as theta:
            json.dump(self.theta,theta,indent=2,ensure_ascii=False)

        with open(self.parm.ResultFile + '/ModelResult/phi.txt','w',encoding='utf-8') as phi:
            for k in range(self.K):
                for v in range(self.V):
                    phi.write(str(self.phi[k][v]) + ';')

                phi.write('\n')

        with open(self.parm.ResultFile + '/ModelResult/psi.txt','w',encoding='utf-8') as psi:
            for item in self.psi:
                json.dump(item,psi,ensure_ascii=False)
                psi.write('\n')

        with open(self.parm.ResultFile + '/ModelResult/tassign.json','w',encoding='utf-8') as tassign:
            j = 0
            for doc in self.fenciinf:
                i = 0
                num = doc[3]
                for word in doc[0]:
                    self.tassign[num].append((self.word2id[word], self.word[j][i], self.inventor[j][i]))
                    i += 1
                j += 1
            json.dump(self.tassign,tassign,indent=2,ensure_ascii=False)

        with open(self.parm.ResultFile + '/ModelResult/tau.json','w',encoding='utf-8') as tau:
            json.dump(self.tau,tau,indent=2,ensure_ascii=False)

        logging.info("结果保存完成")

    #保存中间结果
    def middle_save(self,iternum):
        with open(self.parm.ResultFile + '/Middle/word' + str(iternum) + '.txt','w',encoding='utf-8') as w:
            for m in range(self.M):
                w.write(self.fenciinf[m][3] + ':')
                for n in range(self.doclength[self.fenciinf[m][3]]):
                    w.write(str(self.word[m][n]) + '-' + str(self.inventor[m][n]) + ';')

                w.write('\n')

        with open(self.parm.ResultFile + '/Middle/nw'+ str(iternum) + '.txt','w',encoding='utf-8') as nw:
            for k in range(self.K):
                for v in range(self.V):
                    nw.write(str(self.nw[k][v]) + ';')

                nw.write('\n')

            for k in range(self.K):
                nw.write(str(self.nwsum[k]) + ';')

            nw.write('\n')

        with open(self.parm.ResultFile + '/Middle/m'+ str(iternum) + '.json','w',encoding='utf-8') as m:
            json.dump(self.m,m,ensure_ascii=False)
            m.write('\n')
            json.dump(self.msum,m,ensure_ascii=False)

        with open(self.parm.ResultFile + '/Middle/nd'+ str(iternum) + '.txt','w',encoding='utf-8') as nd:
            for m in range(self.M):
                nd.write(self.fenciinf[m][3] + ':')
                for k in range(self.K):
                    nd.write(str(self.nd[m][k]) + ';')

                nd.write('\n')

            for m in range(self.M):
                nd.write(str(self.ndsum[m]) + ';')
            nd.write('\n')

        with open(self.parm.ResultFile + '/Middle/nc'+ str(iternum) + '.txt','w',encoding='utf-8') as nc:
            for k in range(self.K):
                json.dump(self.nc[k],nc,ensure_ascii=False)
                nc.write('\n')

            for k in range(self.K):
                nc.write(str(self.ncsum[k]) + ';')
            nc.write('\n')

        logging.info("第" + str(iternum) + "次迭代结果保存完成")

    #读取中途存储的文件到参数中
    def read_middle(self,topicnum,iternum,date):
        with open('../SaveOldResult/'+ str(topicnum) + '-' + date + '/Middle/word' + str(iternum) + '.txt','r',encoding='utf-8') as w:
            docs = w.readlines()
            num = []
            words = []
            for m in range(len(docs)):
                num.append(docs[m].strip('\n').split(':')[0])
                words.append(docs[m].strip('\n').split(':')[1].split(';')[:-1])

            for m in range(self.M):
                index = num.index(self.fenciinf[m][3])
                for n in range(self.doclength[num[index]]):
                    self.word[m][n] = int(words[index][n].split('-')[0])
                    self.inventor[m][n] = words[index][n].split('-')[1]

        print('word.txt读取完成')

        with open('../SaveOldResult/'+ str(topicnum) + '-' + date + '/Middle/nw'+ str(iternum) + '.txt','r',encoding='utf-8') as nw:
            topics = nw.readlines()
            for k in range(self.K):
                words = topics[k].strip('\n').split(';')[:-1]
                for v in range(self.V):
                    self.nw[k][v] = int(words[v])

        print('nw.txt')

        with open('../SaveOldResult/'+ str(topicnum) + '-' + date + '/Middle/m'+ str(iternum) + '.json','r',encoding='utf-8') as m:
            authors = m.readlines()
            self.m = json.loads(authors[0])
            self.msum = json.loads(authors[1])

        print('m.json')

        with open('../SaveOldResult/'+ str(topicnum) + '-' + date + '/Middle/nd'+ str(iternum) + '.txt','r',encoding='utf-8') as nd:
            docs = nd.readlines()
            num = []
            topics = []
            total = docs[-1].split(';')[:-1]
            for doc in docs[:-1]:
                m = doc.strip('\n').split(':')
                num.append(m[0])
                topics.append(m[1].split(';')[:-1])

            for m in range(self.M):
                index = num.index(self.fenciinf[m][3])
                for k in range(self.K):
                    self.nd[m][k] = int(topics[index][k])
                self.ndsum[m] = int(total[index])

        print('nd.txt')

        with open('../SaveOldResult/'+ str(topicnum) + '-' + date + '/Middle/nc'+ str(iternum) + '.txt','r',encoding='utf-8') as nc:
            companies = nc.readlines()
            ncsum = companies[-1].split(';')[:-1]
            for k in range(self.K):
                self.nc[k] = json.loads(companies[k])
                self.ncsum[k] = int(ncsum[k])

        print('nc.txt')

        logging.info("参数文件读取成功")

    #训练新增数据
    def new_data(self):
        with open('./result/tau'+ str(self.K) + '.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            i = 0
            for line in lines:
                self.tau.append([])
                words = line.split('    ')[:-1]
                for word in words:
                    self.tau[i].append(float(word))
                i += 1

        with open('./result/theta'+ str(self.K) + '.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            i = 0
            for line in lines:
                self.theta.append([])
                words = line.split('    ')[:-1]
                for word in words:
                    self.theta[i].append(float(word))
                i += 1

        with open('./result/phi'+ str(self.K) + '.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            i = 0
            for line in lines:
                self.phi.append([])
                topics = line.split('    ')[:-1]
                for topic in topics:
                    self.phi[i].append(float(topic))
                i += 1

        with open('./result/psi'+ str(self.K) + '.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            i = 0
            for line in lines:
                self.psi.append([])
                words = line.split('    ')[:-1]
                for word in words:
                    self.psi[i].append(float(word))

def func(client, patentinf, word, inventor, nd, nw, nwsum, m, msum, nc, ncsum):
    return client.process(patentinf, word, inventor, nd, nw, nwsum, m, msum, nc, ncsum)

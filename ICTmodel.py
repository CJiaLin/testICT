#2.0版ICT模型
import numpy as np
import random
from tqdm import tqdm
import logging


logging.basicConfig(level=logging.DEBUG,
                    filename='logging.log',
                    filemode='a',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s-%(message)s')
logger = logging.getLogger(__name__)

class ICT():
    M = 0         #文档数
    K = 0         #Topic数
    V = 0         #不重复词的个数
    A = 0         #发明者人数
    C = 0         #公司数
    iternum = 0   #迭代次数

    word = []     #存储第m篇文档的第n个词被指定的Topic id   M*per_doc_word_len
    inventor = []   #存储第m篇文档的第n个词指定的发明者ID     M*per_doc_word_len
    nd = []       #第i篇文档中第j个Topic产生的词的个数   M*K
    ndsum = []    #第i篇文档的总词数   M
    nw = []       #第i个Topic中第j个词的个数   K*V
    nwsum = []    #第i个Topic的总词数    K
    m = []        #第i个发明人下Topic j出现次数  A*K
    msum = []     #第i个发明人的所有主题出现次数和 A
    nc = []       #公司i中主题j出现的次数     C*K
    ncsum = []    #公司i中所有主题出现的次数总和  C

    fenciinf = []  #文档分词结果
    doclength = [] #每篇文档的总词数

    theta = []    #inventor-topic 二维矩阵，存储每个inventor下Topic的概率 A*K
    phi = []      #topic-word 二维矩阵，存储每个Topic下Word的概率    K*V
    psi = []      #company-topic 二维矩阵，存储每个company下Topic的概率   C*K
    tau = []      #patent-topic 二维矩阵，存储每个Patent下Topic的概率

    word2id = {}
    id2word = {}

    alpha = 0.0
    beta = 0.0
    mu = 0.0
    omega = 0.0


    def __init__(self,topicnum,iternum,length_all,length_single,fenciinf,word2id,inventor2id,company2id):
        self.K = int(topicnum)
        self.A = len(inventor2id)
        self.C = len(company2id)
        self.iternum = int(iternum)
        self.alpha = 50.0/self.K
        self.beta = 0.01
        self.mu = 0.01
        self.doclength = length_all
        self.V = length_single
        self.word2id = word2id
        self.inventor2id = inventor2id
        self.company2id = company2id
        self.fenciinf = []
        self.savenum = 100
        self.omega = 0.01

        self.id2word = {v:k for k,v in self.word2id.items()}

        for patent in fenciinf:
            self.fenciinf.append([patent[0]+patent[1],patent[2],patent[3]])
        self.M = len(self.fenciinf)
        try:
            self.creat_value()
            logging.info("创建数据成功！")

            self.init_value()
            logging.info("初始化数据成功！")
        except Exception as e:
            logging.error("数据初始化失败！", exc_info=True)

        self.iter()


    def creat_value(self):
        for i in range(self.M):
            self.word.append([])
            self.inventor.append([])
            self.nd.append([])
            self.tau.append([])
            for j in range(self.doclength[i]):
                self.word[i].append(-1)
                self.inventor[i].append(-1)

            for j in range(self.K):
                self.nd[i].append(0)
                self.tau[i].append(0.0)

            self.ndsum.append(0)

        for i in range(self.K):
            self.nw.append([])
            self.nwsum.append(0)
            self.phi.append([])
            self.nc.append([])
            self.ncsum.append(0)
            self.psi.append([])

            for j in range(self.V):
                self.nw[i].append(0)
                self.phi[i].append(0.0)

            for j in range(self.A):
                if i == 0:
                    self.m.append([])
                    self.msum.append(0)
                    self.theta.append([])

                self.m[j].append(0)
                self.theta[j].append(0.0)

            for j in range(self.C):
                self.nc[i].append(0)
                self.psi[i].append(0.0)

    #初始化数据
    def init_value(self):
        for m in range(self.M):
            for n in range(self.doclength[m]):
                topic_id = random.randint(0,self.K-1)     #选取主题
                ran_inv = random.randint(0,len(self.fenciinf[m][1])-1)
                inventor_id = self.inventor2id[self.fenciinf[m][1][ran_inv]]
                word_id = self.word2id[self.fenciinf[m][0][n]]

                self.word[m][n] = topic_id
                self.inventor[m][n] = inventor_id

                self.nw[topic_id][word_id] += 1
                self.nwsum[topic_id] += 1
                self.nd[m][topic_id] += 1
                self.ndsum[m] += 1

                self.m[inventor_id][topic_id] += 1
                self.msum[inventor_id] += 1

                for company in self.fenciinf[m][2]:
                    self.nc[topic_id][self.company2id[company]] += 1
                    self.ncsum[topic_id] += 1


    #迭代word-topic分布情况，同时计算theta,phi,psi
    def iter(self):
        logging.info("开始迭代")
        for i in tqdm(range(self.iternum)):
            try:
                for m in range(self.M):
                    for n in range(self.doclength[m]):
                        topic_id = self.word[m][n]
                        inventor_id = self.inventor[m][n]
                        word_id = self.word2id[self.fenciinf[m][0][n]]

                        self.nw[topic_id][word_id] -= 1
                        self.nwsum[topic_id] -= 1
                        self.nd[m][topic_id] -= 1

                        self.m[inventor_id][topic_id] -= 1
                        self.msum[inventor_id] -= 1

                        for company in self.fenciinf[m][2]:
                            self.nc[topic_id][self.company2id[company]] -= 1
                        self.ncsum[topic_id] -= 1

                        newtopic,newinventor = self.probability(m, n)

                        self.word[m][n] = newtopic
                        self.inventor[m][n] = newinventor

                        self.nw[newtopic][word_id] += 1
                        self.nwsum[newtopic] += 1
                        self.nd[m][newtopic] += 1

                        self.m[newinventor][newtopic] += 1
                        self.msum[newinventor] += 1

                        for company in self.fenciinf[m][2]:
                            self.nc[newtopic][self.company2id[company]] += 1
                        self.ncsum[newtopic] += 1

            except Exception as e:
                logging.error("第"+str(i)+"次迭代失败", exc_info = True)

        for x in range(self.A):
            for k in range(self.K):
                self.theta[x][k] = (self.m[x][k] + self.alpha) / (self.msum[x] + self.K * self.alpha)
        logging.info("theta计算完成")
        for k in range(self.K):
            for v in range(self.V):
                self.phi[k][v] = (self.nw[k][v] + self.beta) / (self.nwsum[k] + self.V * self.beta)
        logging.info("phi计算完成")
        for c in range(self.C):
            for k in range(self.K):
                self.psi[k][c] = (self.nc[k][c] + self.mu) / (self.ncsum[k] + self.C * self.mu)
        logging.info("psi计算完成")

        for m in range(self.M):
            for k in range(self.K):
                self.tau[m][k] = (self.nd[m][k] + self.omega) / (self.ndsum[m] + self.K * self.omega)
        logging.info('omega计算完成')

    #计算每个Topic的概率
    def probability(self,m,n):
        Kalpha = self.K * self.alpha
        Vbeta = self.V * self.beta
        Cmu = self.C * self.mu
        word_id = self.word2id[self.fenciinf[m][0][n]]
        topic_id = 0

        p = []
        for inventor in self.fenciinf[m][1]:
            inventor_id = self.inventor2id[inventor]
            for k in range(self.K):
                company_p = 0.0
                for company in self.fenciinf[m][2]:
                    company_p += (self.nc[k][self.company2id[company]] + self.mu) / (self.ncsum[k] + Cmu)

                p.append(((self.m[inventor_id][k] + self.alpha) / (self.msum[inventor_id] + Kalpha)) * ((self.nw[k][word_id] + self.beta) / (self.nwsum[k] + Vbeta)) * (company_p / len(self.fenciinf[m][2])))

        for j in range(1,len(p)):
            p[j] += p[j-1]

        u = np.random.rand() * p[len(p)-1]

        for i in range(len(p)):
            if p[i] > u:
                topic_id = i
                break

        inventor = int(topic_id / self.K)
        newtopic = topic_id - inventor * self.K
        newinventor = self.inventor2id[self.fenciinf[m][1][inventor]]

        return newtopic,newinventor

    #保存结果
    def save(self):
        with open('result/theta'+str(self.K)+'.txt','w') as theta:
            for x in range(self.A):
                for k in range(self.K):
                    theta.write(str(self.theta[x][k]) + '    ')

                theta.write('\n')

        with open('result/phi'+str(self.K)+'.txt','w') as phi:
            for k in range(self.K):
                for v in range(self.V):
                    phi.write(str(self.phi[k][v]) + '    ')

                phi.write('\n')

        with open('result/psi'+str(self.K)+'.txt','w') as psi:
            for k in range(self.K):
                for c in range(self.C):
                    psi.write(str(self.psi[k][c]) + '    ')

                psi.write('\n')

        with open('result/tassign'+str(self.K)+'.txt','w') as tassign:
            for m in range(self.M):
                for n in range(self.doclength[m]):
                    w = self.fenciinf[m][0][n]
                    tassign.write(str(self.word2id[w]) + ':' + str(self.word[m][n]) + '    ')

                tassign.write('\n')

        with open('result/tau'+str(self.K)+'.txt','w') as tau:
            for m in range(self.M):
                for k in range(self.K):
                    tau.write(str(self.tau[m][k]) + '    ')

                tau.write('\n')

        logging.info("结果保存完成")

    #保存中间结果
    def middle_save(self,iternum):
        with open('middle/word'+str(self.K)+'.txt','w') as w:
            for m in range(self.M):
                for n in range(self.doclength[m]):
                    w.write(str(self.word[m][n]) + ':' + str(self.inventor[m][n]) + '    ')

                w.write('\n')

        with open('middle/nw'+str(self.K)+'.txt','w') as nw:
            for k in range(self.K):
                for v in range(self.V):
                    nw.write(str(self.nw[k][v]) + '    ')

                nw.write('\n')

            for k in range(self.K):
                nw.write(str(self.nwsum[k]) + '    ')

            nw.write('\n')

        with open('result/m'+str(self.K)+'.txt','w') as m:
            for a in range(self.A):
                for k in range(self.K):
                    m.write(str(self.m[a][k]) + '    ')

                m.write('\n')

            for a in range(self.A):
                m.write(str(self.msum[a]) + '    ')
            m.write('\n')

        with open('result/nd'+str(self.K)+'.txt','w') as nd:
            for m in range(self.M):
                for k in range(self.K):
                    nd.write(str(self.nd[m][k]) + '    ')

                nd.write('\n')

            for m in range(self.M):
                nd.write(str(self.ndsum[m]) + '    ')
            nd.write('\n')

        with open('result/nc'+str(self.K)+'.txt','w') as nc:
            for k in range(self.K):
                for c in range(self.C):
                    nc.write(str(self.nc[k][c]) + '    ')

                nc.write('\n')

            for k in range(self.K):
                nc.write(str(self.ncsum[k]) + '    ')
            nc.write('\n')

        logging.info("第" + str(iternum) + "次迭代结果保存完成")
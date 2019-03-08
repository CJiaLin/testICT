import datetime
import os
import logging
import json
from pymongo import MongoClient
from sqlalchemy import create_engine, Column,String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.types import DateTime
from sqlalchemy.interfaces import PoolListener
import pandas as pd
from ast import literal_eval
from tqdm import tqdm

logger = logging.getLogger('main.Parm')

Base = declarative_base()

class MyListener(PoolListener):
    def connect(self, dbapi_con, con_record):
        dbapi_con.execute("PRAGMA temp_store = 2")

class DB(Base):
    __tablename__ = 'patent'
    __table_args__ = {'extend_existing': True}

    id = Column(String, primary_key=True)
    category = Column(String)
    pub_number = Column(String)
    app_number = Column(String)
    pub_date = Column(DateTime)
    title = Column(String)
    abstract = Column(String)
    description = Column(String)
    claims = Column(String)
    inventor = Column(String)
    assignee = Column(String)

    def to_dict(self):
        d = {}
        d['id'] = self.id
        d['category'] = self.category
        d['pub_date'] = self.pub_date
        d['title'] = self.title
        d['abstract'] = self.abstract
        d['description'] = self.description
        d['claims'] = self.claims
        d['inventor'] = self.inventor
        d['assignee'] = self.assignee
        d['cited_patents'] = [c.cited_pat for c in self.cited_patents]
        return d

    def __repr__(self):
        return "<Patent(id='%s', category='%s', pub_number='%s', app_number='%s',\
                        pub_date='%s', title='%s', abstract='%s', description='%s',\
                        claims='%s', cited_patents='%s')>" %(self.id, self.category,
                        self.pub_number, self.app_number, str(self.pub_date), self.title,
                        self.abstract, self.description, self.claims, self.cited_patents)

class GetInf():
    def __init__(self):
        self.patentinf = {}
        self.testinf = []
        self.titles = {}
        conn = MongoClient('mongodb://210.45.215.233:9001/')
        #210.45.215.233:9001   localhost:27017
        self.db = conn['big_patent']      #big_patent

    def CHbyDB(self):
        set = self.db['patent_meta']       #patent_meta
        for file in set.find({'address':{'$regex' : ".*安徽省.*"}}):
            # patentinf.append([abstract[i], inventor, company])
            self.patentinf[str(file['_id'])] = {'title': file['patent_name'],
                                           'abstract': file['abstract'],
                                           'sovereignty': file['sovereignty'],
                                           'inventor': file['inventor'],
                                           'company': file['applicant'],
                                            'type': file['main_type_number']}

        with open('../Document/big_patent.json','w',encoding='utf-8') as patent:
            json.dump(self.patentinf,patent,ensure_ascii=False,indent=2)

        logger.info("专利数据读取成功！")

    def CHbyJSON(self):
        #patent = []
        #with open('../Document/big_patent.json', 'r', encoding='utf-8') as patent:
            #self.patentinf = json.load(patent)
        with open('../Document/Anhui.txt','r',encoding='utf-8') as patent:
            lines = patent.readlines()
            for line in lines:
                inf = json.loads(line)
                if inf['patent_type'] != '外观设计':
                    self.patentinf[str(inf['public_number'])] = {'title':inf['patent_name'],
                                                                 'abstract': inf['abstract'],
                                                                 'sovereignty': inf['sovereignty'],
                                                                 'inventor': inf['inventor'],
                                                                 'company': inf['applicant']}

        return self.patentinf

    def ENbyDB(self):
        DB_URL = 'sqlite:///E:/机器学习/testICT/Document/patent.db'
        engine = create_engine(DB_URL, listeners=[MyListener()])
        engine.text_factory = str
        #metadata = Base.metadata
        Session = sessionmaker(bind=engine)
        session = Session()

        pat_id = []
        #inventor = []
        patent = session.query(DB.id).all()
        #print('2')
        #patent = session.query(DB.id).filter(DB.inventor == '[]').all()
        #print('1')
        i = 0
        for pat_id in tqdm(patent):
            pat = session.query(DB).filter(DB.id == pat_id[0])[0]
            try:
                inv = ",".join([inv.decode('utf-8') for inv in literal_eval(pat.inventor)])
                ass = ",".join([ass.decode('utf-8') for ass in literal_eval(pat.assignee)])
            except Exception:
                inv = pat.inventor
                ass = pat.assignee

            pat.inventor = inv
            pat.assignee = ass
            if i >= 2000:
                session.commit()
                i = 0
            else:
                i += 1

        session.commit()
        '''with open('../Document/fai2.txt','w') as f:
            for pat in tqdm(patent):
                #print(pat)
                inventor = session.query(DB.inventor).filter(DB.id == pat[0]).first()
                if inventor[0] == "[]" or inventor[0] == None:
                    f.write(pat[0] + '\n')'''

        '''with open('../Document/fai2.txt','r') as f:
            pat_id = f.readlines()'''
            #inventor.append(pat.inventor)
            #self.patentinf[str(pat.id)] = {'title': pat.title,
            #                               'abstract': pat.abstract,
            #                               'claims': pat.claims}
        return pat_id,session

    def ENbyCSV(self):
        csv_url = "../Document/patent.csv"
        csv = pd.read_csv(csv_url)
        pat_ids = csv.id
        titles = csv.title
        abstract = csv.abstract
        sovereignty = csv.claims
        inventor = csv.inventor
        company = csv.assignee
        i = 0
        for id in tqdm(pat_ids):
            self.patentinf[str(id)] = {'title': titles[i],
                                        'abstract': abstract[i],
                                        'sovereignty': sovereignty[i].strip(),
                                        'inventor': inventor[i].split(','),
                                        'company': str(company[i]).split(',')}
                                        #'inventor': [inv.decode('utf-8') for inv in literal_eval(inventor[i])],
                                        #'company': [com.decode('utf-8') for com in literal_eval(company[i])]}
            i += 1
        return csv


class Parm():
    now_time = datetime.datetime.now().strftime('%m-%d')
    def __init__(self,topicnum,date=None):
        self.core = 64
        self.savenum = 100
        self.StopWordCH = 'stopword.txt'
        self.StopWordEN = 'stopword_en.txt'
        self.Dict = 'userdict_AI.txt'
        self.DocFielPath = '../Document/'
        if date != None:
            self.now_time = date
            self.ResultFile = '../SaveOldResult/' + str(topicnum) + '-' + self.now_time
        else:
            self.ResultFile = '../SaveOldResult/' + str(topicnum) + '-' + self.now_time
            #self.WordResultPath = '/WordResult/'
            #self.MiddleFilePath = '/Middle/'
            #self.MatchFilePath = '/MatchResult/'
            #self.ModelResultPath = '/ModelResult/'
            if not os.path.exists('../SaveOldResult/'+ str(topicnum) + '-' +self.now_time ):
                os.makedirs('../SaveOldResult/' + str(topicnum) + '-' + self.now_time)
            if not os.path.exists('../SaveOldResult/' + str(topicnum) + '-' + self.now_time + '/MatchResult'):
                os.makedirs('../SaveOldResult/' + str(topicnum) + '-' + self.now_time + '/MatchResult')
            if not os.path.exists('../SaveOldResult/' + str(topicnum) + '-' + self.now_time + '/WordResult'):
                os.makedirs('../SaveOldResult/' + str(topicnum) + '-' + self.now_time + '/WordResult')
            if not os.path.exists('../SaveOldResult/' + str(topicnum) + '-' + self.now_time + '/ModelResult'):
                os.makedirs('../SaveOldResult/' + str(topicnum) + '-' + self.now_time + '/ModelResult')
            if not os.path.exists('../SaveOldResult/' + str(topicnum) + '-' + self.now_time + '/Middle'):
                os.makedirs('../SaveOldResult/' + str(topicnum) + '-' + self.now_time + '/Middle')

    def GetWordResult(self,patentinf):
        inventor2id = []
        company2id = []

        with open(self.ResultFile + '/WordResult/length.json', 'r', encoding='utf-8') as lenfile:
            length_all = json.load(lenfile)

        with open(self.ResultFile + '/WordResult/wordmap.json', 'r', encoding='utf-8') as wordmap:
            word2id = json.load(wordmap)
            length_single = len(word2id)

        '''with open(self.ResultFile + '/WordResult/inventorsmap.json', 'w', encoding='utf-8') as inventors:
            json.dump(inventor2id,inventors,ensure_ascii=False,indent=2)

        with open(self.ResultFile + '/WordResult/companymap.json', 'w', encoding='utf-8') as companies:
            json.dump(company2id,companies,ensure_ascii=False,indent=2)'''

        with open(self.ResultFile + '/WordResult/fenci_final.json', 'r', encoding='utf-8') as fencifile:
            fenci = json.load(fencifile)
            for num,doc in fenci.items():
                patentinf[num]['fenci'] = doc['title'] + doc['content']

        with open(self.ResultFile + '/WordResult/inventor.txt','r',encoding='utf-8') as inventor:
            lines = inventor.readlines()
            for item in lines:
                inventor2id.append(item.strip())

        with open(self.ResultFile + '/WordResult/company.txt','r',encoding='utf-8') as company:
            lines = company.readlines()
            for item in lines:
                company2id.append(item.strip())

        return length_all,length_single,patentinf,word2id,inventor2id,company2id
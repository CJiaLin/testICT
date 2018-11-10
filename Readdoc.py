#!/usr/bin/python
#coding=utf-8
import os
from Duedoc import Duedoc
#from DuedocByTFIDF import Duedoc
from ICTmodel import ICT
#from rewriterICT import ICT
import pandas as pd
from perplexity import per
import operator
import logging
from ResultTest import resulttest


logging.basicConfig(level=logging.DEBUG,
                    filename='logging.log',
                    filemode='a',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s-%(message)s')
logger = logging.getLogger(__name__)

word2id = {}
length = []
M = 0
K = 0

def readdoc():
    filepath = './document'
    files = os.listdir(filepath)
    patentinf = []
    testinf = []
    titles = []
    abstract = []
    #M = len(files)
    length_single = 0
    #提取文件中专利的简介、公司、发明人信息
    '''for file in files:
        if not os.path.isdir(file):
            xlsx_f = xlrd.open_workbook(filepath+'\\'+file)
            ALL = xlsx_f.sheet_by_name('Sheet1')
            abstract = ALL.col_values(4)
            inventors = ALL.col_values(13)
            companies = ALL.col_values(12)
            for i in range(len(abstract)):
                inventor = inventors[i].split(',')
                company = companies[i].split(',')
                patentinf.append([abstract[i],inventor,company])'''

    for file in files:
        if not os.path.isdir(file):
            f = open(filepath+'/'+file,encoding='utf-8')
            csv_f = pd.read_csv(f)
            date = csv_f['AD']
            abstract = csv_f['ABS']
            inventors = csv_f['INV']
            companies = csv_f['PA']
            title = csv_f['TI']
            for i in range(len(abstract)):
                inventor = inventors[i].split(';')[:-1]
                company = companies[i].split(';')[:-1]
                #patentinf.append([abstract[i], inventor, company])
                if date[i][0:4] != '2017':
                    patentinf.append([title[i],abstract[i],inventor,company])
                    titles.append(title[i])
                else:
                    testinf.append([title[i],inventor,company])


    logger.info("专利数据读取成功！")
    return patentinf,testinf,titles,abstract

def duedoc(patentinf):
    doc = Duedoc(patentinf)
    #del(patentinf)
    #doc.due()
    doc.duefenci()
    del doc

def saveinf(patentinf,topicnum):
    inventor2id = {}
    company2id = {}
    length_all = []
    i = 0
    j = 0
    for items in patentinf:
        for item in items[2]:
            if item not in inventor2id:
                inventor2id[item] = i
                i += 1

        for item in items[3]:
            if item not in company2id:
                company2id[item] = j
                j += 1

    with open('word_result/length.txt', 'r', encoding='utf-8') as lenfile:
        lines = lenfile.readlines()
        for i in range(len(lines)):
            length_all.append(int(lines[i].strip('\n')))

    with open('word_result/wordmap.txt', 'w', encoding='utf-8') as wordmap:
        with open('word_result/count.txt', 'r', encoding='utf-8') as count:
            lines = count.readlines()
            length_single = len(lines)
            for i in range(length_single):
                wordmap.write(lines[i].split(':')[0] + ':' + str(i) + '\n')
                word2id[lines[i].split(':')[0].strip('\n')] = i

    with open('word_result/inventorsmap.txt','w',encoding='utf-8') as inventors:
        for inventor,num in inventor2id.items():
            inventors.write(inventor + ':' + str(num) + '\n')

    with open('word_result/companymap.txt','w',encoding='utf-8') as companies:
        for company, num in company2id.items():
            companies.write(company + ':' + str(num) + '\n')

    with open('word_result/fenci_final.txt','r',encoding='utf-8') as fenci:
        lines = fenci.readlines()
        for i in range(len(lines)):
            patentinf[i][0] = lines[i].split('----')[0].split(';')[:-1]
            patentinf[i][1] = lines[i].split('----')[1].split(';')[:-1]

    ict = ICT(topicnum,1000,length_all,length_single,patentinf,word2id,inventor2id,company2id)
    #ict = HierarchicICT(topicnum, 1000, length_all, length_single, patentinf, word2id, inventor2id, company2id)
    ict.save()

def outpu(topicnum):
    result = []
    result_sort = []
    with open('word_result/wordmap.txt', 'r', encoding='utf-8') as wordmap:
        lines = wordmap.readlines()
        for line in lines:
            word2id[line.strip('\n').split(':')[0]] = line.strip('\n').split(':')[1]

    id2word = {int(v): k for k, v in word2id.items()}
    with open('result/phi'+str(topicnum)+'.txt','r', encoding='utf-8') as phifile:
        lines = phifile.readlines()
        i = 0
        for line in lines:
            result.append({})
            words = line.split('    ')[:-1]
            #print('topic' + str(i) + ':    ', end='')
            j = 0
            for word in words:
                result[i][id2word[j]] = float(word)
                #print(id2word[j] + ':' + str(word) + '    ', end='')
                j += 1
            #print('\n')
            i += 1

    i = 0
    with open('result/top30'+str(topicnum)+'.txt','w') as top:
        for item in result:
            top.write('topic' + str(i) + ':    ')
            print('topic' + str(i) + ':    ', end='')
            result_sort.append(sorted(item.items(), key=operator.itemgetter(1), reverse=True))
            for content in result_sort[i][0:31]:
                print(content[0] + '\t',end='')
                top.write(content[0] + "\t")
            top.write('\n')
            print('\n',end='')
            i += 1

    #print(topic)

def cal_perp(topicnum,patentinf):
    inventor2id = {}
    company2id = {}
    id2word = {}
    with open('word_result/inventorsmap.txt','r',encoding='utf-8') as inventorsmap:
        lines = inventorsmap.readlines()
        for line in lines:
            inventor2id[line.split(':')[0]] = int(line.split(':')[1].strip('\n'))

    with open('word_result/companymap.txt','r',encoding='utf-8') as companymap:
        lines = companymap.readlines()
        for line in lines:
            company2id[line.split(':')[0]] = int(line.split(':')[1].strip('\n'))

    with open('word_result/wordmap.txt', 'r', encoding='utf-8') as wordmap:
        lines = wordmap.readlines()
        for line in lines:
            id2word[int(line.split(':')[1])] = line

    perp = per(topicnum,patentinf,inventor2id,company2id)
    perp_result = perp.calculate()

    return perp_result

def test(topicnum,patentinf,word):
    inventor2id = {}
    company2id = {}
    with open('word_result/inventorsmap.txt','r',encoding='utf-8') as inventorsmap:
        lines = inventorsmap.readlines()
        for line in lines:
            inventor2id[line.split(':')[0]] = int(line.split(':')[1].strip('\n'))

    with open('word_result/companymap.txt','r',encoding='utf-8') as companymap:
        lines = companymap.readlines()
        for line in lines:
            company2id[line.split(':')[0]] = int(line.split(':')[1].strip('\n'))

    with open('word_result/fenci_final.txt','r',encoding='utf-8') as fenci:
        lines = fenci.readlines()
        for i in range(len(lines)):
            patentinf[i][0] = lines[i].split('----')[0].split(';')[:-1]
            patentinf[i][1] = lines[i].split('----')[1].split(';')[:-1]

    result = resulttest(patentinf,word2id,inventor2id,company2id,word[:100],topicnum)
    #result.count()
    result.topic_probability()
    topdoc = result.select()

    return topdoc

if __name__ == "__main__":
    topicnum = 10
    #topicnum = [5,10,15,20,25,30,35]
    perp = []
    patentinf,testinf,titles,abstract = readdoc()
    #duedoc(patentinf)
    #saveinf(patentinf,topicnum)
    #outpu(topicnum)

    #del patentinf,testinf,titles,abstract
    with open('word_result/wordmap.txt', 'r', encoding='utf-8') as wordmap:
        lines = wordmap.readlines()
        for line in lines:
            word2id[line.split(':')[0]] = int(line.split(':')[1])

    topdoc = test(topicnum, patentinf, testinf)
    result = []
    name = ['testtitle','top1','top2','top3','top4','top5','top6','top7','top8','top9','top10']

    i = 0
    for patent in topdoc:
        result.append([])
        result[i].append(testinf[i][0])
        for num in patent:
            result[i].append(str(num) + ':' + titles[num] + '----' + abstract[num])
        i += 1

    csv = pd.DataFrame(columns=name, data=result)

    csv.to_csv('./result/topdoc' + str(topicnum) + '.csv')

    logging.info("程序结束")
from ICTModelParallel import ICT
#from Duedoc import DueEnDoc
import json
import operator
from Duedoc import Duedoc
from Parm import Parm    #各种文件路径的参数
from Parm import GetInf
from ResultTest import resulttest
#import Experiment
import sys
import logging

logging.basicConfig(level=logging.DEBUG,
                    filename='logging.log',
                    filemode='a',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s-%(message)s')
logger = logging.getLogger('main')

GetDoc = GetInf()
Lang = 'CH'
#Lang = 'EN'
#inf_type = 'DB'
#inf_type = 'CSV'
inf_type = 'json'

Doc = None
patentinf = None


def GetTopic(topicnum,patentinf,date):
    PatentTopic = {}
    with open('../SaveOldResult/'+ str(topicnum) + '-' + date +  '/ModelResult/phi.txt','r',encoding='utf-8') as tau:
        lines = tau.readlines()
        for i in range(len(lines)):
            patent = lines[i].split('    ')[:-1]
            PatentTopic[patentinf[i][4]] = patent.index(max(patent))

    with open('./MatchResult/PatentTopic' + str(topicnum) + '.json','w',encoding='utf-8') as pt:
        json.dump(PatentTopic,pt)

def TopWord(topicnum,date):
    result = []
    result_sort = []
    with open('../SaveOldResult/'+ str(topicnum) + '-' + date +  '/WordResult/wordmap.json', 'r', encoding='utf-8') as wordmap:
        word2id = json.load(wordmap)

    id2word = {int(v): k for k, v in word2id.items()}
    with open('../SaveOldResult/'+ str(topicnum) + '-' + date +  '/ModelResult/phi.txt','r', encoding='utf-8') as phifile:
        lines = phifile.readlines()
        i = 0
        for line in lines:
            result.append({})
            words = line.split(';')[:-1]
            j = 0
            for word in words:
                result[i][id2word[j]] = float(word)
                j += 1
            i += 1

    i = 0
    with open('../SaveOldResult/'+ str(topicnum) + '-' + date +  '/MatchResult/wordrank.txt','w',encoding='utf-8') as top:
        for item in result:
            top.write('topic' + str(i) + ':')
            print('topic' + str(i) + ':', end='')
            result_sort.append(sorted(item.items(), key=operator.itemgetter(1), reverse=True))
            for content in result_sort[i][0:300]:
                print(content[0] + ';',end='')
                top.write(content[0] + ";")
            top.write('\n')
            print('\n',end='')
            i += 1

    result = []
    result_sort = []
    with open('../SaveOldResult/'+ str(topicnum) + '-' + date +  '/ModelResult/psi.txt','r', encoding='utf-8') as psifile:
        lines = psifile.readlines()
        for line in lines:
            result.append(json.loads(line))
    i = 0
    with open('../SaveOldResult/'+ str(topicnum) + '-' + date +  '/MatchResult/companyrank.txt','w',encoding='utf-8') as top:
        for item in result:
            top.write('topic' + str(i) + ':')
            print('topic' + str(i) + ':', end='')
            result_sort.append(sorted(item.items(), key=operator.itemgetter(1), reverse=True))
            for content in result_sort[i][0:300]:
                print(content[0] + ';',end='')
                top.write(content[0] + ";")
            top.write('\n')
            print('\n',end='')
            i += 1

if __name__ == '__main__':
    topicnum = 45
    if Lang == 'CH' and inf_type == 'DB':
        patentinf = GetDoc.CHbyJSON()
        GetDoc.CHbyDB()
    elif Lang == 'CH' and inf_type == 'json':
        patentinf = GetDoc.CHbyJSON()
    elif Lang == 'EN' and inf_type == 'DB':
        patentinf = GetDoc.ENbyDB()
        sys.exit()
    elif Lang == 'EN' and inf_type == 'CSV':
        patentinf = GetDoc.ENbyCSV()
    print('============程序选择==============')
    print('=============1.分词===============')
    print('=============2.模型===============')
    print('==========3.分词+模型=============')
    print('=========4.从中断点继续===========')
    print('=====5.每个Topic的Top300词========')
    print('==6.每个inventor、Company的Top10==')
    print('===========7.实验结果=============')
    print('=============8.退出===============')
    select = input('输入选择的步骤：')
    if select == '1':
        Parm = Parm(topicnum)
        Doc = Duedoc(patentinf.copy(),topicnum)
        Doc.due()
        Doc.duefenciBySubSampling()
    elif select == '2':
        Parm = Parm(topicnum)
        length_all,length_single,patent,word2id,inventor2id,company2id = Parm.GetWordResult(patentinf.copy())
        Model = ICT(topicnum,1000,length_all,length_single,patent,word2id,inventor2id,company2id,Parm)
        Model.save()
    elif select == '3':
        Parm = Parm(topicnum)
        Doc = Duedoc(patentinf.copy(),topicnum)
        Doc.due()
        Doc.duefenciBySubSampling()
        length_all,length_single,patent,word2id,inventor2id,company2id = Parm.GetWordResult(patentinf.copy())
        Model = ICT(topicnum,1000,length_all,length_single,patent,word2id,inventor2id,company2id,Parm)
        Model.save()
    elif select == '4':
        date = input('中间结果的保存日期：')
        iternum = int(input('从第几次迭代开始：'))
        Parm = Parm(topicnum,date=date)
        length_all,length_single,patent,word2id,inventor2id,company2id = Parm.GetWordResult(patentinf.copy())
        Model = ICT(topicnum,1000-iternum,length_all,length_single,patent,word2id,inventor2id,company2id,Parm,middle=iternum)
        Model.read_middle(topicnum,iternum,date)
        Model.iter()
        Model.save()
    elif select == '5':
        date = input('保存日期（月-日）：')
        TopWord(topicnum, date)
    elif select == '6':
        date = input('保存日期（月-日）：')
        Parm = Parm(topicnum,date=date)
        length_all,length_single,patent,word2id,inventor2id,company2id = Parm.GetWordResult(patentinf.copy())
        Result = resulttest(patentinf.copy(),word2id,inventor2id,company2id,topicnum,date)
        Result.Patent()
        #Result.Company()
        #Result.Inventor()
    elif select == '7':
        topicnum = input('主题数：')
        date = input('保存日期（月-日）：')
        pass
    else:
        pass


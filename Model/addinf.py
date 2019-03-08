from Parm import GetInf
from bs4 import BeautifulSoup as bsoup
import random
from urllib import request
from Parm import DB
from urllib.request import urlopen
from urllib.error import URLError
from tqdm import tqdm
import gevent.monkey
gevent.monkey.patch_all()
#import threading

def get_patent(patent_id='US20110221676'):
    # make the google patent link
    link = 'https://patents.google.com/patent/%s/de' % patent_id
    #link = 'https://www.google.de/patents/%s?cl=de' % patent_id
    # get the html of the webpage containing the original article
    inv = []
    ass = []
    try:
        # random magic browser so you don't immediately get blocked out for making too many requests ;-)
        #ssl._create_default_https_context = ssl._create_unverified_context
        #gcontext = ssl.SSLContext()
        #sess = requests.Session()
        #sess.mount('https://', DESAdapter())
        #html = sess.get(link,headers={'User-Agent': 'Magic Browser%i'%random.randint(0,100)})
        #print(html.text)
        #time.sleep(3)
        html = urlopen(request.Request(link, headers={'User-Agent': 'Magic Browser%i'%random.randint(0,100)})).read()
    except URLError as e:
        try:
            html = urlopen(request.Request(link, headers={'User-Agent': 'Magic Browser%i'%random.randint(0,100)})).read()
        except Exception as a:
            with open('../Document/fai.txt', 'a') as f:
                f.write(patent_id + '\n')
            print('未知错误1：', e)
            print(link)
            return patent_id,str(inv),str(ass)
    except Exception as e:
        with open('../Document/fai.txt', 'a') as f:
            f.write(patent_id + '\n')
        print('未知错误2：', e)
        print(link)
        return patent_id,str(inv),str(ass)
    # make a beautiful soup object out of the html
    soup = bsoup(html,features="html.parser")
    # extract patent title
    try:
        inventors = soup.findAll('dd', {'itemprop': 'inventor'})
        assignees = soup.findAll('dd', {'itemprop': 'assigneeCurrent'}) # .get_text(separator=u' ').encode('utf-8').strip()
        if inventors != []:
            for inventor in inventors:
                try:
                    inv.append(inventor.get_text(separator=u' ').encode('utf-8').strip())
                except Exception as e:
                    print(str(patent_id) + "专利获取发明人失败")
                    print(e)

        if assignees != []:
            for assignee in assignees:
                try:
                    ass.append(assignee.get_text(separator=u' ').encode('utf-8').strip())
                except:
                    print(str(patent_id) + "专利获取公司失败")
    except Exception as e:
        print(str(patent_id) + "利获取公司失败:")
        print(e)

    inv = str(inv)
    ass = str(ass)
    return patent_id,inv,ass

if __name__=='__main__':
    with open('../Document/fai.txt','w') as f:
        f.write('')
    getinf = GetInf()
    patent, session = getinf.ENbyDB()
    length = len(patent)
    #Pool = Process
    p = 50
    part = int(length/p)
    inventor = []
    assignee = []
    #pi,inv,ass = get_patent('US20010005513')
    #print(inv,ass)
    i = 0
    '''for pat in tqdm(patent):
        pat_id, inv, ass = get_patent(pat.strip())
        pat = session.query(DB).filter(DB.id == pat_id)[0]
        pat.inventor = inv
        pat.assignee = ass
    session.commit()'''
    for i in tqdm(range(p)):
        jobs = []
        if i != p-1:
            for pat in patent[i*part:(i+1)*part]:
                jobs.append(gevent.spawn(get_patent,pat.strip()))
            gevent.joinall(jobs)
        else:
            for pat in patent[i*part:]:
                jobs.append(gevent.spawn(get_patent,pat.strip()))
            gevent.joinall(jobs)

        for i,g in enumerate(jobs):
            pat_id,inv,ass = g.value
            pat = session.query(DB).filter(DB.id == pat_id)[0]
            pat.inventor = inv
            pat.assignee = ass
            #print(inv,ass)
        session.commit()

    quit()
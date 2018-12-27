import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

process = 30
#patents = pd.read_csv('document/train_all_19887+3601(149).tsv', sep='\t', header=None, error_bad_lines=False)
patents = pd.read_csv('document/patent.tsv', sep='\t',error_bad_lines=False)
inventor = pd.read_csv('document/inventor.tsv', sep='\t', error_bad_lines=False, index_col=0)
assignee = pd.read_csv('document/assignee.tsv', sep='\t', error_bad_lines=False, index_col=0)

#print(patents)
patent_number = patents['number']
patent_date = patents['date']
patent_abstract = patents['abstract']
patent_title = patents['title']
#patent_claim = patents['claim']
del patents
patent_length = len(patent_number)

patent_inventors = pd.read_csv('document/patent_inventor.tsv', sep='\t', error_bad_lines=False, low_memory=False, index_col=0)
patent_assignees = pd.read_csv('document/patent_assignee.tsv', sep='\t', error_bad_lines=False, low_memory=False, index_col=0)


def InCo(patent_number,patent_date):
    inventors = []
    companies = []
    for i in tqdm(range(len(patent_number))):
        inf_inventor = ''
        inf_company = ''
        try:
            if int(patent_date[i][:4]) >= 2001 and int(patent_date[i][:4]) <= 2012:
                inventor_id = patent_inventors.ix[str(patent_number[i])]  # 可能有多个，返回一个DataFrame，如果是单个则返回Series
                if type(inventor_id).__name__ == 'Series':
                    item = inventor_id['inventor_id']
                    if len(inventor.ix[str(item)]['name_first'].split('\n')) <= 1:
                        inf_inventor += inventor.ix[str(item)]['name_last'] + ' ' + inventor.ix[str(item)]['name_first'] + ';'
                else:
                    for item in inventor_id['inventor_id']:
                        inf_inventor += inventor.ix[str(item)]['name_last'] + inventor.ix[str(item)]['name_first'] + ';'
        except KeyError as e:
            pass
        except Exception as e:
            print('Inventor ERROR:', e)
        inventors.append(inf_inventor)

        try:
            if int(patent_date[i][:4]) >= 2001 and int(patent_date[i][:4]) <= 2012:
                company_id = patent_assignees.ix[str(patent_number[i])]
                if type(company_id).__name__ == 'Series':
                    if str(assignee.ix[company_id['assignee_id']]['organization']) != 'nan':
                        inf_company += assignee.ix[str(company_id['assignee_id'])]['organization'] + ';'
                    else:
                        inf_company += assignee.ix[str(company_id['assignee_id'])]['name_last'] + ' ' + assignee.ix[str(company_id['assignee_id'])]['name_first'] + ';'
                else:
                    for item in company_id['assignee_id']:
                        if str(assignee.ix[str(item)]['organization']) != 'nan':
                            inf_company += str(assignee.ix[str(item)]['organization']) + ';'
                        else:
                            inf_company += str(assignee.ix[str(item)]['name_last']) + ' ' + str(assignee.ix[str(item)]['name_first']) + ';'
        except KeyError as e:
            pass
        except Exception as e:
            print('Company ERROR:', e)
        companies.append(inf_company)

    return inventors,companies


if __name__ == '__main__':
    patent_inf = []

    inf_inventor = []
    inf_company = []

    # inventors,companies = InCo(patent_number[:100])
    # inf_inventor += inventors
    # inf_company += companies

    p = Pool(process)

    result = []

    for i in range(process):
        if i != process - 1:
            inf = list(patent_number[i * int(patent_length / process):(i + 1) * int(patent_length / process)])
            result.append(p.apply_async(InCo, args=(inf,patent_date[i * int(patent_length / process):(i + 1) * int(patent_length / process)])))
        else:
            inf = list(patent_number[i * int(patent_length / process):])
            result.append(p.apply_async(InCo, args=(inf,patent_date[i * int(patent_length / process):])))

    p.close()
    p.join()

    for res in result:
        inventors, companies = res.get()
        inf_inventor += inventors
        inf_company += companies

    for i in tqdm(range(patent_length)):
        if inf_inventor[i] != '' and inf_company[i] != '':
            patent_inf.append([patent_number[i], patent_title[i], patent_abstract[i], inf_inventor[i],inf_company[i]])

    name = ['number', 'title', 'abstract', 'inventor', 'company']
    csv = pd.DataFrame(columns=name, data=patent_inf)
    csv.to_csv('./document/patent_inf.csv')

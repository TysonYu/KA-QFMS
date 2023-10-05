from factsumm import FactSumm
import jsonlines

factsumm = FactSumm()

with jsonlines.open('/home/tiezheng/workspace/FidSum/data/qmsum/processed_data_no_sent_split/test.jsonl') as F:
    article = []
    target = []
    for item in F:
        article.append(item['src'])
        target.append(item['tgt'])

with open('src.txt', 'w') as F:
    F.writelines([item+'\n' for item in article])

with open('tgt.txt', 'w') as F:
    F.writelines([item+'\n' for item in target])

# with open('/home/tiezheng/workspace/FidSum/data/qmsum/test_results/12_512_no_KA_in_generation_penalty1_maxlen256.results') as F:
    
#     summary = F.readlines()
#     summary = [item.strip('\n') for item in summary]


# # factsumm(article[0], summary[0], device="cuda")

# article = "Lionel Andrés Messi (born 24 June 1987) is an Argentine professional footballer who plays as a forward and captains both Spanish club Barcelona and the Argentina national team. Often considered as the best player in the world and widely regarded as one of the greatest players of all time, Messi has won a record six Ballon d'Or awards, a record six European Golden Shoes, and in 2020 was named to the Ballon d'Or Dream Team."
# summary = "Lionel Andrés Messi (born 24 Aug 1997) is an Spanish professional footballer who plays as a forward and captains both Spanish club Barcelona and the Spanish national team."
# factsumm(article, summary, verbose=True, device="gpu")


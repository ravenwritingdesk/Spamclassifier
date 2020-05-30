from os.path import exists
import pickle
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import SparseMatrixSimilarity
from numpy import argsort
#import wordChopping

#读取unmarked文件

#unmarked_i文件在trashClassifier.py中可选择生成,此时已带有label
with open('./result/unmarked_i.txt', encoding='utf-8') as f1:
        texts1=f1.read().split('\n')

#读取marked文件
with open('./data/marked.txt', encoding='utf-8') as f2:
        texts=f2.read().split('\n')


#合并文件集合
texts=texts+texts1

# TF-IDF模型实现
PATH = 'model.pickle'#保存到pickle中，故一旦生成pickle文件，后续查询花费时间极小
if exists(PATH):
    with open(PATH, 'rb') as f:
        dictionary, tfidf = pickle.load(f)#若pickle文件存在则load
else:
    corpora = [list(t) for t in texts]
    #建立词典
    dictionary = Dictionary(corpora)
    tfidf = TfidfModel(dictionary.doc2bow(c) for c in corpora)
    #写入词典和pickle
    with open(PATH, 'wb') as f:
        pickle.dump((dictionary, tfidf), f)#dump

# 搜索并rank，输出一定数量的top
num_features = len(dictionary.token2id)
while True:#实现循环查询
    kw = input('输入查询的内容：').strip()
    kw_vec = dictionary.doc2bow(list(kw))#vector_kw
    texts_kw = [t for t in texts if kw in t]#遍历查询
    corpora_kw = [dictionary.doc2bow(list(t)) for t in texts_kw]
    index = SparseMatrixSimilarity(tfidf[corpora_kw], num_features)
    sim = index[tfidf[kw_vec]]  # TF-IDF搜索
    ids = argsort(-sim)[:100] #返回前100
    for i in ids:
        print(texts_kw[i])


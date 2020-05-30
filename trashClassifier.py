import pandas as pd
import jieba
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import  MultinomialNB
from sklearn.metrics import accuracy_score

#读取标记数据
data = pd.read_csv(r"./data/marked.txt", sep='\t', names=['label', 'text'])

#jieba分词
data['cut_message'] = data["text"].apply(lambda x: ' '.join(jieba.cut(x)))

#value数据集
x = data['cut_message'].values
y = data['label'].values

#分割训练集和数据集
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1)  # test_size:train_size=1:9

#向量化
vectorizer = CountVectorizer()
x_train_termcounts = vectorizer.fit_transform(train_x)

tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(x_train_termcounts)

#开始训练
classifier = MultinomialNB().fit(x_train_tfidf, train_y)

#向量化
x_input_termcounts = vectorizer.transform(test_x)
x_input_tfidf = tfidf_transformer.transform(x_input_termcounts)

#预测
predicted_categories = classifier.predict(x_input_tfidf)

#测试其分类精度
print(accuracy_score(test_y, predicted_categories))
print(metrics.classification_report(test_y, predicted_categories))
print(metrics.confusion_matrix(test_y, predicted_categories))


'''
#把unmark数据变为mark数据，便于搜索引擎输出标签
data2 = pd.read_csv(r"./data/unmarked.txt", sep='\n', names=['text'])
data2['cut_message'] = data2["text"].apply(lambda x: ' '.join(jieba.cut(x)))
x_input_termcounts2= vectorizer.transform(data2['cut_message'])
x_input_tfidf2 = tfidf_transformer.transform(x_input_termcounts2)
predicted_categories2 = classifier.predict(x_input_tfidf2)
k=open('./result/unmarked_i.txt', 'w', encoding='UTF-8')
#print(predicted_categories2)
f = open("./data/unmarked.txt","r")
for line in f:
    list(predicted_categories2).append(line)
for n in  list(predicted_categories2):
    k.write(n)
f.close()
k.close()
'''
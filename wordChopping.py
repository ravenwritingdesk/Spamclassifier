import jieba

#unmark文件分词
#打开raw文件
file1 = open('./data/unmarked.txt', 'r', encoding='UTF-8')
#读取
content = file1.read()
#jieba分词
cut_content = jieba.cut(content)
#打开result文件
file1_r = open('./result/unmarked.txt', 'w', encoding='UTF-8')
#写入
file1_r.write(' '.join(cut_content))
#关闭两文件
file1.close()
file1_r.close()

#mark文件分词
#打开raw文件
file2 = open('./data/marked.txt', 'r', encoding='UTF-8')
#读取
content2 = file2.read()
#jieba分词
cut_content2 = jieba.cut(content2)
#打开result文件
file2_r = open('./result/marked.txt', 'w', encoding='UTF-8')
#写入
file2_r.write(' '.join(cut_content2))
#关闭两文件
file2.close()
file2_r.close()


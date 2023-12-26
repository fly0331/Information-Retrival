
import math
from nltk.stem import PorterStemmer
import re
from stop_words import get_stop_words

class TF_IDF():
    def __init__(self):
        self.path = "./data"
        self.num_docs = 1095 #文章數
        self.token_list = [[] for i in range(self.num_docs+1)] #token_list記錄每篇文章tokenization後的term
        self.allwords = []
        self.stopword = []
        self.Tokenization() # step1 先將所有文章進行tokenization
        self.tf = []
        self.df = {}
        self.idf = {}
        self.tfidf_unit_vector = []
        self.bow = {}
        self.cal_tfidf() #step 2 計算tfidf

    #<---------Stemming------------->
    def Stemming(self, tokens):
        ps = PorterStemmer()
        word = []
        for token in tokens:
            word.append(ps.stem(token))
        return word

    #<---------remove stop words------------->
    def removeStopWord(self, str):
        
        stopwords = get_stop_words('en')
        text = ' '.join([word for word in str.split() if word not in stopwords])
        
        return(text)

    #<---------Tokenization------------->
    # Tokenization會呼叫2個functions
    # 1.removeStopWord 2.Stemming
    def Tokenization(self):
        
        #處理從第1篇~第1095篇文章
        for i in range(1,self.num_docs+1):
            #讀檔
            file_path = self.path+"/"+str(i)+".txt"
            f = open(file_path,'r')
            doc = f.read()
            
            doc = re.sub('\n', ' ', doc) #移除換行符號
            doc = re.sub('[^A-Za-z\']+', ' ', doc) #只留下英文&'的字元
            doc = doc.lower() #將所有英文字元都轉為小寫
            
            filtered_string = self.removeStopWord(doc) #移除stopwords
            filtered_string = re.sub('\'', ' ', filtered_string) #再把還有"'"的地方清掉
            filtered_string = re.sub(r'\b\w{1}\b', ' ', filtered_string) #把有些被濾到只剩一個char的字串刪掉
            filtered_string = re.sub(' +', ' ', filtered_string) #把有連續>=2個white space的地方改成一格就好
            filtered_string = filtered_string.strip() #把文章前後的空白刪掉
            token = filtered_string.split(' ') #以空白鍵來分割文字成token
            self.token_list[i] = self.Stemming(token) #用Porter’s algorithm 來進行Stemming
            
     
    """
    計算tf,idf結果
    tf:[{word1:3,word2:4,word4:2},{word2:5,word3:7, word4:2},{....},.......]
    df:{word1:{df:6個doc, t_index:1},word2:{df:3個doc, t_index:2},word3:{df:5個doc, t_index:3},word4:{df:4個doc, t_index:4}......}
    idf:{word1:idf(word1),word2:idf(word2),word3:idf(word3)..........}
    """
    def cal_tfidf(self):
        #處理從第1篇~第1095篇文章
        for i in range(1,self.num_docs+1):
            bow = {} #bow為暫存doc[i]所有term的term frequency ex.bow:{word1:5, word2:7,...}
            for word in self.token_list[i]: #遍歷doc[i]其token_list 計算每個word在doc[i]的出現次數
                if not word in bow: 
                    bow[word] = 0
                bow[word] += 1
            self.tf.append(bow) #加到tf中，tf以List方式記錄每個doc的term freq.
            for word in bow.keys():  #遍歷bow.keys()(也就是doc[i]的set(token_list[i])) 計算每個word在所有doc中 總共出現在幾篇doc
                if word not in self.df:
                    self.df[word] = {}
                    self.df[word]['df'] = 0
                self.df[word]['df'] += 1
        self.df = dict(sorted(self.df.items())) #將df依term排序好
        #計算df裡的term其idf值 idf = log10(N/df)
        for word in self.df.keys(): 
            self.idf[word] = math.log10(self.num_docs / self.df[word]['df'])
            
    def tf(self, index, word):
        return self.tf[index-1][word]
     
    def idf(self, word):
        return self.idf[word]
 
    def tf_idf(self, index, word):
        return self.tf[index-1][word]*self.idf[word]


    def cosine_similarity(self, v1, v2):
        sum = 0
    
        for x_id in v1.keys():
            if x_id in v2.keys():
                sum+= v1[x_id]*v2[x_id]
        return sum
        



    
tf_idf = TF_IDF()


# (1) Construct a dictionary
path = './dictionary.txt'
f = open(path, 'w')
row = 0
print("{:<8} {:<12} {:<8}".format('t_index','term','df'), file = f) # print 欄位名
for term in tf_idf.df.keys(): #遍歷df中所有的term
    row += 1 
    tf_idf.df[term]['t_index'] = row # row為t_index
    print("{:<8} {:<12} {:<8}".format(row, term, tf_idf.df[term]['df']), file = f) #寫到dictionary.txt
            
# (2) Transfer each document into a tf-idf unit vector.   

        
for i in range(1, tf_idf.num_docs+1): #處理從第1篇~第1095篇文章
    
            
    tf_id_list = [] 
    tfidf_list = []
    length = 0 #計算長度用，為了將tfidf轉為單位向量
   
    for term in sorted(set(tf_idf.token_list[i])) : #遍歷doc[i]的set(token_list[i]))
        t_index = tf_idf.df[term]['t_index'] #取出dictionary.txt中的term其t_index
        tfidf = tf_idf.tf_idf(i, term) #計算此term在doc[i]的tfidf值
        tf_id_list.append(t_index) 
        tfidf_list.append(tfidf)
        length += tfidf * tfidf
    # 轉為unit vector  tfidf_unit_vector:[{2:0.025, 3:0.004, ...},{1:0.001, 5:0.147, ...}, ...]      
    tf_idf.tfidf_unit_vector.append({tf_id_list[j]: tfidf_list[j]/ math.sqrt(length) for j in range(len(tf_id_list))})
    
    #write to output
    path = './output'+"/"+str(i)+".txt"
    f = open(path, 'w')
    print("{:<8} {:<8}".format('t_index','tf-idf'), file = f) # print 欄位名
    for t_index, tfidf in tf_idf.tfidf_unit_vector[i-1].items(): # -1因為單位向量從index=0開始存
        print("{:<8} {:<8}".format(t_index, tfidf), file = f) #寫到./output/[i].txt
        
        
# (3) returns cosine similarity of DocX and DocY 
print("請輸入你想計算哪兩篇Documents的cosine similarity")
x = int(input('第一篇(輸入一個數字)：'))
y = int(input('第二篇(輸入一個數字)：'))
tfidf_x = tf_idf.tfidf_unit_vector[x-1]
tfidf_y = tf_idf.tfidf_unit_vector[y-1]  

print(tf_idf.cosine_similarity(tfidf_x, tfidf_y))    


import math
from nltk.stem import PorterStemmer
import re
from stop_words import get_stop_words
import csv

class TF_IDF():
    def __init__(self, path, num_docs):
        self.path = path
        self.num_docs = num_docs #文章數
        self.token_list = [[] for i in range(self.num_docs+1)] #token_list記錄每篇文章tokenization後的term
        self.allwords = []
        self.stopword = []
        self.Tokenization() # step1 先將所有文章進行tokenization
        self.tf = [{0}]
        self.df = {}
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
    計算tf,df結果
    tf:[{word1:3,word2:4,word4:2},{word2:5,word3:7, word4:2},{....},.......]
    df:{word1:{df:6個doc, t_index:1},word2:{df:3個doc, t_index:2},word3:{df:5個doc, t_index:3},word4:{df:4個doc, t_index:4}......}
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
    
class NaiveBayes():
    def __init__(self, tf_idf:TF_IDF, classes:list, training_dataset:list, training_list:dict):
        self.tf_idf = tf_idf
        self.classes = classes
        self.training_dataset = training_dataset
        self.training_list = training_list
        
    def TrainMultinomialNB(self, V:list) -> list | dict :
        #宣告prior陣列存放各類別的P(C)值
        prior = [0 for i in range(0, len(self.classes)+1)]
        condprob = {}
        N = len(self.training_dataset)
        
        for c in self.classes:
            Nc = CountDocsInClass(c)
            prior[c] = Nc/N
            text_c = ConcatenateTextOfAllDocsInClass(self.training_list[c])
            
            num_of_term_in_class_c = sum(list(text_c.values()))

            for term in V:
                
                T_ct = CountTokensOfTerm(text_c, term)
                if term not in condprob:
                    condprob[term] = [1/(num_of_term_in_class_c+len(V)) for i in range(0, len(self.classes)+1)]
                
                condprob[term][c] = (T_ct+1)/(num_of_term_in_class_c+len(V))
        return prior, condprob


    def ApplyMultinomialNB(self, V:list, categories:list, prior:list, condprob:dict, d:int) -> int:
        word_in_d = ExtractTokensFromDoc(d)
        score = [0 for i in range(0, len(categories)+1)]
        max_score = -10000000
        mapping = 0
        for c in categories:
            score[c] = math.log(prior[c])
            for term in word_in_d:
                #只計算selected_features裡的字，其他不用算分數
                if term in V:
                    score[c] += math.log(condprob[term][c]) * tf_idf.tf[d][term]
            #紀錄最高分的類別是哪個
            if score[c] > max_score:
                max_score = score[c]
                mapping = c
        return mapping

    

##########################################################################################         
#計算text_c中的query_term出現次數
def CountTokensOfTerm(text_c:dict, query_term:str) -> int:
    times = 0
    for term in text_c.keys():
        if query_term == term:
            times += text_c[term]
            break
    return times
    
#將category c中所有docs的term合併，並計算tf in these docs
#   term_in_c:{word1:3,word2:4,word4:2,...} 
def ConcatenateTextOfAllDocsInClass(to_read_docs:list) -> dict:
    term_in_c = {}
    for i in to_read_docs:
        for term in ExtractTokensFromDoc(i):
            if term not in term_in_c:
                term_in_c[term] = tf_idf.tf[i][term]
            else:
                term_in_c[term] += tf_idf.tf[i][term]
    return term_in_c
    
#算訓練集裡的文章總數
def CountDocsInClass(c:int) -> int:
    return len(training_list[c])

#回傳doc d的term set
def ExtractTokensFromDoc(d:int) -> list:
    return tf_idf.tf[d].keys()
    
    
#取出字典裡的term
def ExtractVocabulary(D:list) -> list:
    V = []
    for doc in D:
        V.extend(ExtractTokensFromDoc(doc))
    V = set(V)
    return V
    
  
def likelihood_ratio(c:int, V:list) -> dict:
    score = {}
    for term in V:
        n11 = 0
        n01 = 0
        n10 = 0
        n00 = 0
        for doc in training_dataset:
            #若此doc為此類別(on topic)
            if doc in training_list[c]:
                #若term為present
                if term in ExtractTokensFromDoc(doc):
                    n11+=1
                #若term為absent
                else:
                    n10+=1
            #若此doc為不屬於此類別(off topic)
            else:
                #若term為present
                if term in ExtractTokensFromDoc(doc):
                    n01+=1
                #若term為absent
                else:
                    n00+=1
        pt = (n11+n01)/len(training_dataset)
        p1 = n11/(n11+n10)
        p2 = n01/(n01+n00)
        H1_likelihood = (math.pow(pt,n11)) * (math.pow((1-pt),n10)) * (math.pow(pt,n01)) * (math.pow((1-pt),n00))
        H2_likelihood = (math.pow(p1,n11)) * (math.pow((1-p1),n10)) * (math.pow(p2,n01)) * (math.pow((1-p2),n00))
        llr = (-2)*(math.log(H1_likelihood)-math.log(H2_likelihood))
        score[term] = llr
    return score
    
def FeaturesSelection(training_list:dict,classes:list, k:int) -> list:
    vocabulary = []
    score_list = {} #用來儲存每個class中的每個term的llr score e.g.score_list={'apple':0.025,'banana':3.521,...}
    #針對各個class各挑k個重要features最後合併起來
    for c in classes:
        V = ExtractVocabulary(training_list[c])
        score_list = likelihood_ratio(c, V)
        #將score_list依照values(llr score)排序
        sorted_score = dict(sorted(score_list.items(), key=lambda item: item[1], reverse=True))
        #每個class挑最高分的k個features合併起來
        vocabulary.extend(list(sorted_score.keys())[:k])
            
    return vocabulary
    
    

# step1:藉由HW2先根據所有data建立dictionary&tfidf table
file_path = "./data"
num_docs = 1095
tf_idf = TF_IDF(file_path, num_docs)
        
        
training_dataset = [] #用list來儲存所有要拿來training的doc id
training_list = {} #用dict分別儲存每個class中有那些training的doc id
classes = [] #用list來儲存class有哪些
f = open('training.txt')
for line in f.readlines():
    input = line.split(' ')
    if '\n' in input:
        input.remove('\n')
    input = [int(i) for i in input]
    training_list[input[0]] = input[1:]
    training_dataset.extend(input[1:])    
    classes.append(input[0])
f.close
training_dataset.sort()
        
# step2: 根據training dataset先做重要term的篩選
selected_features = FeaturesSelection(training_list, classes, int(500/13))
selected_features = set(selected_features)
    
# step3: 用Naive Bayes進行training
NB = NaiveBayes(tf_idf, classes, training_dataset, training_list)
prior, condprob = NB.TrainMultinomialNB(selected_features)
        
# step4: 將剩下的資料集進行apply 
mapping_class_of_doc = [0 for i in range(1, num_docs+2)]
for doc in range(1,num_docs+1):   
    mapping_class_of_doc[doc] = NB.ApplyMultinomialNB(selected_features, classes, prior, condprob, doc)
            


# step5: write result to  'HW3_csv.csv'      
header = ['Id','Value']
data = []
for i in range(1,num_docs+1):
    if i not in training_dataset:
        data.append([i,mapping_class_of_doc[i]])


with open('HW3_csv.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    # write multiple rows
    writer.writerows(data)

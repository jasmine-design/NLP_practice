import re
import numpy as np
import pandas as pd
import jieba.posseg as pseg
import jieba
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load custom dictionary
file_path = 'custom_dict.txt'
jieba.load_userdict(file_path)

# Read the Excel file
df = pd.read_excel('GoogleNews_2022.12.29.xlsx', sheet_name='Sheet1', engine='openpyxl')

# Replace comma with period as sentence delimiter
sentences = df['paragraph'].str.replace(r'\s+', '').str.replace(',', '.')

# Split the text into sentences using period as delimiter
sentences = sentences.str.split('.')

word_list = []
for i, sentence in enumerate(sentences):
    # 遍歷分割後的子句
    for sub_sentence in sentence:
        # 去除字串前後的空格
        sub_sentence = sub_sentence.strip()
        # 判斷是否為空字串
        if sub_sentence:
            # 使用pseg.cut()斷詞並取得詞性標注結果
            words = pseg.cut(sub_sentence)
            # 將斷詞及詞性標注結果加入 word_list 中
            word_list += [sub_sentence for word in words]

n_topics = 10   ### 分幾個topics
n_top_words = 150   ### 顯示topic中多少個字(關鍵字)

tf_vectorizer = CountVectorizer(token_pattern='[\u4e00-\u9fff]{2,6}',max_features=500)
tf = tf_vectorizer.fit_transform(word_list)
lda = LatentDirichletAllocation(n_components=n_topics, max_iter=50,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
lda.fit(tf)

feature_names = tf_vectorizer.get_feature_names_out()
compoments = lda.components_

#匯出主題關鍵字在excel
topic_dic = {}
for no in range(n_topics):
  topic = ([feature_names[i] for i in compoments[no].argsort()[:-n_top_words - 1:-1]])
  topic_dic["topic"+str(no+1)] = topic

topic_df = pd.DataFrame(topic_dic)
topic_df.to_excel("result.xlsx", index=False)
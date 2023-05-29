import jieba
import pandas as pd
import jieba.posseg as pseg
import paddle

# 啟動詞性標注
paddle.enable_static()

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
            #  使用 pseg.cut() 斷詞，取得詞性標注結果，將斷詞及詞性標注結果加入 word_list 中
            words = pseg.cut(sub_sentence, use_paddle=True)
            word_list += [(sub_sentence, word.word, word.flag) for word in words]

# Convert the word_list into a DataFrame
df_word = pd.DataFrame(word_list, columns=['sentence', 'word', 'pos'])

# Save the DataFrame to Excel
df_word.to_excel('output.xlsx', index=False)

import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import networkx as nx

player_list = ["陳莉鎔","吳芳妤","陳莉鈞","林書荷","陳佳蔓","甘可慧","陳潔","Beatriz De Carvalho",
               "廖苡任","蔡沁瑤","Gabriela Fabiano","陳怡伶","王育文","邱詩晴","賴湘程",## 台北女子鯨華
               "胡曉佩","李怡萱","曾琬羚","李諭","郭覲儀","陳姿雅","黃芯瑜","李姿瑩","許菀芸",
               "洪家瑤","彭𦀡媃","劉煜淳","許惠瑩","廖嘉蓉","黃瀞萱", ## 高雄台電女子
               "葉軒岑","欒妤雯","丁柔安","陳怡如","高歆媛","江思萳","蔡幼群","吳詠涵",
               "陳菀婷","Carla Santos","黃素琴","Aleoscar Blanco","詹佳茵", ## 極速超跑
               "Annerys Vargas Valdez","Dora Peoria","黃情維","亞美娜","林其蓉","童琦芳",
               "胡曉佩","邱雅慧","陳亭汝","劉美菁","陳盈綺","張芷瑄","黃蔓亞","葉於雯","龔詩雯", ##新北中纖女子
               "吳沛緹","鄭伊伶","黃珮甄","洪家瑤","謝靖瑩","洪詩涵","江昀庭","廖嘉安","張稦豈",
               "張洵瑞","胡宸穎","李思依","李子嫣","吳佩茹","潘宣蓉", ##愛山林
               ]

# 讀取 Excel 文件
df = pd.read_excel('output.xlsx')

# 根據 sentence 欄位將詞彙分組並存入列表
word_list = df.groupby('sentence')['word'].apply(list).tolist()

# 使用Word2Vec建立球員的詞嵌入模型
model = Word2Vec(word_list, vector_size=250, window=5, min_count=5, workers=12, epochs=10)
print(model.wv.key_to_index)

# 建立空的相似性矩陣
num_players = len(player_list)
similarity_matrix = [[0] * num_players for _ in range(num_players)]
print(model.wv.key_to_index)

# 計算球員之間的相似性
with open('similarity_player.txt', "w", encoding="utf-8") as f:
    for i in range(num_players):
        for j in range(num_players):
            player1 = player_list[i]
            player2 = player_list[j]
            if player1 not in model.wv.key_to_index or player2 not in model.wv.key_to_index:
                continue
            similarity_matrix[i][j] = cosine_similarity([model.wv[player1]], [model.wv[player2]])[0][0]
            similarity = similarity_matrix[i][j]
            f.write(f"{player1} 和 {player2} 的相似性: {similarity}\n")

# 列出與球員最相關的五個詞彙
with open('similarity_word.txt', 'w', encoding='utf-8') as f:
    for player_name in player_list:
        if player_name not in model.wv.key_to_index:
            continue
        similar_words = model.wv.most_similar(player_name, topn=5)
        f.write(f"與球員 {player_name} 相關的詞彙：\n")
        for word, similarity in similar_words:
            f.write(f"{word}: {similarity}\n")
        f.write('\n')


# 建立相似性圖形
G = nx.Graph()
threshold = 0.8  # 設定相似性閾值
for i in range(num_players):
    for j in range(i + 1, num_players):
        similarity = similarity_matrix[i][j]
        if similarity > threshold:
            G.add_edge(player_list[i], player_list[j], weight=similarity)

# 繪製相似性圖形
pos = nx.spring_layout(G, seed=42)
weights = [G[u][v]['weight']*0.5 for u, v in G.edges()]
colors = [G[u][v]['weight'] for u, v in G.edges()]

# 設定字型
font_path = '/Users/jasmine/Library/Fonts/NotoSansTC-Black.otf'
font_prop = fm.FontProperties(fname=font_path)

# 繪製相似性圖形並設定字型
nx.draw_networkx(G, pos, node_size=5, node_color='lightblue', edge_color='gray', width=weights, font_size=8, font_family=font_prop.get_name())

# 顯示圖形
plt.show()


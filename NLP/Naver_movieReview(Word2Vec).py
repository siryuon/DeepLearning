import urllib.request
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt

urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")

train_data = pd.read_table('ratings.txt')
train_data = train_data.dropna(how='any')
train_data['document'] = train_data['document'].str.replace('[^ㄱ-ㅎ ㅏ-ㅣ 가-힣]','')
stopwords = ['의', '가', '은', '이', '는', '들', '걍', '잘', '좀', '도', '과', '를', '으로', '자', '에', '와', '한', '하다', '로']

okt = Okt()
tokenized_data = []
for sentence in train_data['document']:
    temp_X = okt.morphs(sentence, stem=True)
    temp_X = [word for word in temp_X if not word in stopwords]
    tokenized_data.append(temp_X)


model = Word2Vec(sentences=tokenized_data, size=100, window=5, min_count=5, workers=4, sg=0)

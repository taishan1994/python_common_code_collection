import jieba
import json
from collections import Counter

save_path = "word2id.json"

# 第一步：生成自己数据集的词表
def get_words(texts, save_path, words_num=10000):
  """
  texts：列表，每一个元素是一条文本
  words_num：词表的大小
  """
  words = []
  word2id = {}
  for text in texts:
    word = jieb.lcut(text, cut_all=False)
    words.extend(word)
  words = Counter(words)
  words = sorted(words.items(), key=lambda x:x[1], reverse=True)
  words = words[:words_num]
  words = [i[0] for i in words]
  word2id = {word:i+1 for i, word in enumerate(words)}
  word2id['[PAD]'] = 0
  with open(save_path, 'w', encoding='utf-8') as fp:
    json.dump(word2id, fp, ensure_ascii=False)
  

with open(save_path, 'r') as fp:
  word2id = json.load(fp)


def build_pretrain_embedding(embedding_path, word2id, embedd_dim=100, norm=True):    
    embedd_dict = dict()
    if embedding_path != None:
        # 704368 50
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)

    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([len(word2id), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    pretrain_emb[0,:] = np.random.uniform(-scale, scale, [1, embedd_dim])
    for word, index in word2id.items():
        if word in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index,:] = embedd_dict[word]
            perfect_match += 1
        elif word.lower() in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word.lower()])
            else:
                pretrain_emb[index,:] = embedd_dict[word.lower()]
            case_match += 1
        else:
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrained_size = len(embedd_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s"%(pretrained_size, perfect_match, case_match, not_match, (not_match+0.)/len(word2id)))
    return pretrain_emb, embedd_dim


def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec/root_sum_square

def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r',encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            embedd_dict[tokens[0]] = embedd
    return embedd_dict, embedd_dim
    
 
# 第二步：生成词向量
# embedd_dict, embedd_dim = load_pretrain_emb('./model_hub/ctb.50d.vec')
embedding_path = "ctb.50d.vec"
pretrain_emb, embedd_dim = build_pretrain_embedding(embedding_path, word2id)

print(pretrain_emb.shape, embedd_dim)

# 第三步：在pytorch中可以这么使用
word_embedding = nn.Embedding(pretrain_emb.size(0), embedd_dim)
if pretrain_emb is not None:
    word_embedding.weight.data.copy_(torch.from_numpy(pretrain_emb))
else:
    word_embedding.weight.data.copy_(torch.from_numpy(random_embedding(pretrain_emb.size(0), embedd_dim)))

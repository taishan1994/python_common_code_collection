import torch
import torch.nn.functional as F
from sklearn.preprocessing import normalize
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


def last_token_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


class EmbeddingQwen2Instrcut7B:
    def __init__(self):
        model_dir = "./model_hub/embedding/iic/gte_Qwen2-7B-instruct"
        # from sentence_transformers import SentenceTransformer
        # self.model = SentenceTransformer(model_dir, trust_remote_code=True, device="cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, )
        self.model = AutoModel.from_pretrained(model_dir, trust_remote_code=True, device_map="auto")
        self.model.eval()
        self.model.max_seq_length = 8192
        self.task = 'Given a web search query, retrieve relevant passages that answer the query'

    def encode_text(self, text, prompt_name=None):
        if isinstance(text, str):
            if prompt_name is not None:
                text = [self.task + text]
            else:
                text = [text]
        elif isinstance(text, list):
            if prompt_name is not None:
                text = [self.task + i for i in text]
        batch_dict = self.tokenizer(text, max_length=self.model.max_seq_length, padding=True, truncation=True,
                                    return_tensors='pt')
        batch_dict = batch_dict.to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def cal_similarity(self, text1, text2):
        query_embeddings = self.encode_text(text1, prompt_name="query")
        doc_embeddings = self.encode_text(text2)

        scores = (query_embeddings @ doc_embeddings.T) * 100
        # print(scores.tolist())
        # [[70.00668334960938, 8.184843063354492], [14.62419319152832, 77.71407318115234]]
        return scores.tolist()


class EmbeddingBGE:
    def __init__(self, device="cuda"):
        model_dir = "./model_hub/embedding/AI-ModelScope/bge-large-zh-v1___5"
        # from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_dir, device=device)
        self.instruction = "为这个句子生成表示以用于检索相关文章："

    def encode_text(self, text, prompt_name=None):
        if isinstance(text, str):
            if prompt_name is not None:
                text = [self.instruction + text]
            else:
                text = [text]
        elif isinstance(text, list):
            if prompt_name is not None:
                text = [self.instruction + i for i in text]
        embeddings = self.model.encode(text, normalize_embeddings=True)
        # print(embeddings)
        return embeddings

    def cal_similarity(self, text1, text2):
        query_embeddings = self.encode_text(text1, prompt_name="query")
        doc_embeddings = self.encode_text(text2)

        scores = (query_embeddings @ doc_embeddings.T) * 100
        # print(scores.tolist())
        # [[70.00668334960938, 8.184843063354492], [14.62419319152832, 77.71407318115234]]
        return scores.tolist()


class EmbeddingGTE:
    def __init__(self, device="cuda"):
        model_dir = "./model_hub/embedding/AI-ModelScope/gte-large-zh"
        # from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_dir, device=device)

    def encode_text(self, text):
        if isinstance(text, str):
                text = [text]
        embeddings = self.model.encode(text, normalize_embeddings=True)
        # print(embeddings)
        return embeddings

    def cal_similarity(self, text1, text2):
        query_embeddings = self.encode_text(text1)
        doc_embeddings = self.encode_text(text2)

        scores = (query_embeddings @ doc_embeddings.T) * 100
        # print(scores.tolist())
        # [[70.00668334960938, 8.184843063354492], [14.62419319152832, 77.71407318115234]]
        return scores.tolist()

class EmbeddingXiaobuV2:
    def __init__(self, device="cuda"):
        model_dir = "./model_hub/embedding/models--lier007--xiaobu-embedding-v2/snapshots/ee0b4ecdf5eb449e8240f2e3de2e10eeae877691"
        # from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_dir, device=device)

    def encode_text(self, text):
        if isinstance(text, str):
                text = [text]
        embeddings = self.model.encode(text, normalize_embeddings=True)
        # print(embeddings)
        return embeddings


    def cal_similarity(self, text1, text2):
        query_embeddings = self.encode_text(text1)
        doc_embeddings = self.encode_text(text2)

        scores = (query_embeddings @ doc_embeddings.T) * 100
        # print(scores.tolist())
        # [[70.00668334960938, 8.184843063354492], [14.62419319152832, 77.71407318115234]]
        return scores.tolist()

class EmbeddingStella:
    def __init__(self, device="cuda"):
        model_dir = "./model_hub/embedding/stella-mrl-large-zh-v3.5-1792d"
        # from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_dir, device=device)
        self.dim = 1024

    def encode_text(self, text):
        if isinstance(text, str):
                text = [text]
        embeddings = self.model.encode(text, normalize_embeddings=False)
        embeddings = embeddings[:, :self.dim]
        embeddings = normalize(embeddings)
        # print(embeddings)
        return embeddings

    def cal_similarity(self, text1, text2):
        query_embeddings = self.encode_text(text1)
        doc_embeddings = self.encode_text(text2)

        scores = (query_embeddings @ doc_embeddings.T) * 100
        # print(scores.tolist())
        # [[70.00668334960938, 8.184843063354492], [14.62419319152832, 77.71407318115234]]
        return scores.tolist()

class EmbeddingZpoint:
    def __init__(self, device="cuda"):
        model_dir = "./model_hub/embedding/zpoint_large_embedding_zh"
        # from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_dir, device=device)
        self.dim = 1024

    def encode_text(self, text):
        if isinstance(text, str):
                text = [text]
        embeddings = self.model.encode(text, normalize_embeddings=False)
        embeddings = embeddings[:, :self.dim]
        embeddings = normalize(embeddings)
        # print(embeddings)
        return embeddings

    def cal_similarity(self, text1, text2):
        query_embeddings = self.encode_text(text1)
        doc_embeddings = self.encode_text(text2)

        scores = (query_embeddings @ doc_embeddings.T) * 100
        # print(scores.tolist())
        # [[70.00668334960938, 8.184843063354492], [14.62419319152832, 77.71407318115234]]
        return scores.tolist()

class EmbeddingBCE:
    def __init__(self, device="cuda"):
        model_dir = "./model_hub/embedding/maidalun/bce-embedding-base_v1"
        # from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_dir, device=device)
        # self.dim = 1024

    def encode_text(self, text):
        if isinstance(text, str):
                text = [text]
        embeddings = self.model.encode(text, normalize_embeddings=True)
        # embeddings = embeddings[:, :self.dim]
        # embeddings = normalize(embeddings)
        # print(embeddings)
        return embeddings

    def cal_similarity(self, text1, text2):
        query_embeddings = self.encode_text(text1)
        doc_embeddings = self.encode_text(text2)

        scores = (query_embeddings @ doc_embeddings.T) * 100
        # print(scores.tolist())
        # [[70.00668334960938, 8.184843063354492], [14.62419319152832, 77.71407318115234]]
        return scores.tolist()


if __name__ == '__main__':
    # queries = [
    #     "how much protein should a female eat",
    #     "summit define",
    # ]
    # documents = [
    #     "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    #     "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
    # ]

    queries = [
        "女性每天应该摄入多少蛋白质",
        "峰会的定义",
    ]

    documents = [
        "作为一般指南，CDC建议19至70岁女性的平均蛋白质需求量为每天46克。不过，如你所见，如果你怀孕了或者正在为马拉松训练，你需要增加摄入量。查看下面的图表，了解你每天应该摄入多少蛋白质。",
        "英语学习者的峰会定义：1. 山的最高点；山顶。2. 最高水平。3. 两个或多个政府领导人之间的会议或一系列会议。",
    ]

    # embeddingQwen2Instrcut7B = EmbeddingQwen2Instrcut7B()
    # embeddingQwen2Instrcut7B.cal_similarity(queries, documents)

    embeddingBGE = EmbeddingBGE()
    print(embeddingBGE.cal_similarity(queries, documents))

    embeddingGTE = EmbeddingGTE()
    print(embeddingGTE.cal_similarity(queries, documents))

    embeddingXiaobuV2 = EmbeddingXiaobuV2()
    print(embeddingXiaobuV2.cal_similarity(queries, documents))

    embeddingStella = EmbeddingStella()
    print(embeddingStella.cal_similarity(queries, documents))

    embeddingZpoint = EmbeddingZpoint()
    print(embeddingZpoint.cal_similarity(queries, documents))

    embeddingBCE = EmbeddingBCE()
    print(embeddingBCE.cal_similarity(queries, documents))

    # text = ['xxxx']
    #
    # print(embeddingBGE.encode_text(text, prompt_name="query"))
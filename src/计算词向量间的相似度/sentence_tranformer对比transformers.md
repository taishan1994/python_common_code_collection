# 前言

目前embeddings使用比较多的还是sentence_transformer，一般情况下，使用sentence_transformer加载的模型也可以用huggingface的transformers进行加载推理，这里我们主要是对比一下两者中异同，这里以xiaobu-v2模型为例：

```python
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

xiaobu_path = "/data/gongoubo/szs_pk/model_hub/embedding/models--lier007--xiaobu-embedding-v2/snapshots/ee0b4ecdf5eb449e8240f2e3de2e10eeae877691"

#cona_path = "/data/gongoubo/szs_pk/model_hub/embedding/models--TencentBAC--Conan-embedding-v1/snapshots/fbdfbc53cd9eff1eb55eadc28d99a9d4bff4135f"

#chuxing_path = "/data/gongoubo/szs_pk/model_hub/embedding/models--chuxin-llm--Chuxin-Embedding/snapshots/f4e0b104c8cc2dd9e62c52d7458bd401355b7ea2"

sentence_model = SentenceTransformer(xiaobu_path, device="cuda:0")
```

看下sentence_model分别是什么：

```python
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 1024, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Dense({'in_features': 1024, 'out_features': 1792, 'bias': True, 'activation_function': 'torch.nn.modules.linear.Identity'})
)
```

如果模型权重中包含1_Pooling和2_Dense，那么SentenceTransformer就是由三个部分组成，基础模型、池化模型、Dense模型。其中，基于模型一般是可以用huggingface直接加载的，池化模型是SentenceTransformer中定义的。

我们构造数据获取第0个模型的输出看看：

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained(xiaobu_path)
inputs = tokenizer([text], return_tensors="pt")
output = sentence_model[0](inputs)

"""
{'input_ids': tensor([[ 101,  872, 3221, 6443, 8043,  102]], device='cuda:0'), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0]], device='cuda:0'), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1]], device='cuda:0'), 'token_embeddings': tensor([[[-0.2168,  1.1035, -0.2760,  ...,  0.6899,  1.1266, -0.7204],
         [ 0.0660,  1.1662, -0.0573,  ...,  0.4743,  1.2001, -0.5802],
         [-0.1159,  1.2953, -0.1505,  ...,  0.4793,  1.1762, -0.8607],
         [-0.0528,  1.2045,  0.0244,  ...,  0.4144,  1.1358, -0.6359],
         [-0.1497,  1.2103,  0.1044,  ...,  0.6033,  1.2432, -0.7812],
         [-0.0800,  1.0325, -0.1970,  ...,  0.6916,  1.1420, -0.8119]]],
       device='cuda:0', grad_fn=<NativeLayerNormBackward0>)}
"""
```

然后我们来看下transformers的结果：

```python
transformers_model = AutoModel.from_pretrained(xiaobu_path)
inputs2 = copy.deepcopy(inputs)
transformers_model.eval()
with torch.no_grad():
    output = transformers_model(**inputs2)
    print(output)
    
"""
BaseModelOutputWithPoolingAndCrossAttentions(last_hidden_state=tensor([[[-0.2168,  1.1035, -0.2760,  ...,  0.6899,  1.1266, -0.7204],
         [ 0.0660,  1.1662, -0.0573,  ...,  0.4743,  1.2001, -0.5802],
         [-0.1159,  1.2953, -0.1505,  ...,  0.4793,  1.1762, -0.8607],
         [-0.0528,  1.2045,  0.0244,  ...,  0.4144,  1.1358, -0.6359],
         [-0.1497,  1.2103,  0.1044,  ...,  0.6033,  1.2432, -0.7812],
         [-0.0800,  1.0325, -0.1970,  ...,  0.6916,  1.1420, -0.8119]]]), pooler_output=tensor([[-0.1651,  0.4809,  0.0880,  ..., -0.1331,  0.3532,  0.6633]]), hidden_states=None, past_key_values=None, attentions=None, cross_attentions=None)
"""
```

last_hidden_state和token_embeddings是对应上的。

该输出再接一个池化层，这里直接摘自sentence_transformers里面的部分代码：

```python
token_embeddings = output.last_hidden_state

attention_mask = inputs2.attention_mask
input_mask_expanded = (
    attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
)
sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

sum_mask = input_mask_expanded.sum(1)

sum_mask = torch.clamp(sum_mask, min=1e-9)

pool_output = sum_embeddings / sum_mask

"""
tensor([[-0.0915,  1.1687, -0.0920,  ...,  0.5588,  1.1706, -0.7317]])
"""
```

在这个基础上再接上全连接层：

```python
import torch.nn as nn
dense_model = nn.Linear(1024, 1792, bias=True)
dense_params = torch.load(xiaobu_path + "/2_Dense/pytorch_model.bin", map_location="cpu")
for k,v in dense_params.items():
    if "weight" in k:
        dense_model.weight.data = v
    else:
        dense_model.bias.data = v
        
dense_output = dense_model(pool_output)
print(dense_output)

# 并和sentence_transformers的输出进行对比
embeddings = sentence_model.encode([text], normalize_embeddings=False)
print(embeddings)

"""
tensor([[-0.8267, -1.6052, -1.2769,  ...,  0.2632, -0.7904, -0.8852]])

[[-0.82673246 -1.6051841  -1.276934   ...  0.26318714 -0.7904308
  -0.8852092 ]]
"""
```

最终发现结果是对上了。




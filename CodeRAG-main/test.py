import torch
print(torch.__version__)
# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer("nomic-ai/CodeRankEmbed", trust_remote_code=True)
# queries = ['Represent this query for searching relevant code: Calculate the n-th factorial']
# codes = ['def fact(n):\n if n < 0:\n  raise ValueError\n return 1 if n == 0 else n * fact(n - 1)']
# query_embeddings = model.encode(queries)
# print(query_embeddings)
# code_embeddings = model.encode(codes)
# print(code_embeddings)
# import torch.nn.functional as F
# from transformers import AutoModel, AutoTokenizer

# input_texts = [
#     "how to implement quick sort in Python?",
#     "def quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + middle + quick_sort(right)",
#     "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr",
# ]

# model_path = 'Salesforce/SFR-Embedding-Code-400M_R'
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

# # Tokenize the input texts
# batch_dict = tokenizer(input_texts, max_length=8192, padding=True, truncation=True, return_tensors='pt')

# outputs = model(**batch_dict)
# embeddings = outputs.last_hidden_state[:, 0]

# # normalize embeddings
# embeddings = F.normalize(embeddings, p=2, dim=1)
# scores = (embeddings[:1] @ embeddings[1:].T) * 100
# print("Similarity Scores:", scores.tolist())
# Similarity Scores: [[74.84745025634766, 65.39266967773438]]

# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained(
#     "Salesforce/SFR-Embedding-Code-400M-R",
#     trust_remote_code=True
# )
# 📄 Semantic Document Chunking

This repository provides a method to convert documents into semantically meaningful chunks without relying on fixed chunk sizes or overlapping windows. Instead, it uses semantic chunking, dividing text based on meaning, topics, and the natural structure of the content to preserve contextual relevance.

---

## 🚀 Overview

Traditional chunking techniques split documents based solely on size or fixed length, often leading to fragmented and contextually inconsistent segments.  
Our approach **reverses this by organizing and splitting content based on semantic similarity**, then feeding those into a dynamic chunking strategy. This results in more meaningful and context-aware chunks, and significantly reduces computational costs.

---

## 🔍 How It Works

### 📝 Sentence-wise Splitting
The document is first split into individual sentences or paragraphs, depending on the selected mode (`'sentence'` or `'para'`).

### 🔗 Semantic Segregation

1. Calculate cosine similarity between sentences using a Sentence Transformer.
2. Group sentences where similarity scores > `0.4` into clusters.
3. Recursively repeat for ungrouped sentences until all are grouped semantically.

---

## ⚙️ Dynamic Chunking with Retrieval Optimization

After semantic grouping:
- A **recursive character splitter** is applied with dynamic chunk sizing.
- The chunk size is computed as:

**By default, calling .as_retriever() uses semantic similarity to retrieve the top 4 most relevant chunks.
Typically, one chunk—approximately one-fourth of the document—is enough to provide a meaningful response, depending on the question.**

chunk_size = length_of_document / N
Where N is a configurable parameter that determines granularity.

🔢 Chunk Size Calculation Example
For a document of length 1200 and N = 16:


chunk_size = 1200 / 16 = 75

This would yield chunks of ~75 characters with some overlap.

### 💡 Chunk Usage Guide
Depending on the desired response length, vary how many chunks are used:

| Response Type        | Approx. Chunks Used                       | Chunking Config (`N`, `overlap_ratio`)     |
| -------------------- | ----------------------------------------- | ------------------------------------------ |
| Short Answer         | \~1/4 of total chunks (e.g., top 4 of 16) | `light` → `N=16`, `overlap_ratio=0.15`     |
| Moderate (Detailed)  | \~1/2 of total chunks (e.g., top 6 of 12) | `standard` → `N=12`, `overlap_ratio=0.25`  |
| Detailed Answer      | \~3/4 of total chunks (e.g., top 6 of 8)  | `deep` → `N=8`, `overlap_ratio=0.35`       |
| Very Detailed Answer | All chunks (\~8 of 8)                     | `max_detail` → `N=8`, `overlap_ratio=0.45` |



This balances context coverage and retrieval efficiency.

---

### 🎯 Benefits
✅ Produces contextually relevant and semantically consistent chunks

✅ Saves computational and cost resources by minimizing redundant input

✅ Automatically adjusts chunk size and overlap based on document length and depth

---

### 📦 Installation
Make sure the required dependencies are installed:
```
pip install nltk sentence-transformers langchain
```

If needed, download NLTK tokenizers:
```
import nltk

nltk.download("punkt")
```
### 🧪 Usage Example


```
from splitter import SemanticSplitter

splitter = SemanticSplitter(
    threshold=0.4,                # Semantic similarity threshold for splitting
    depth='standard',            # Options: 'light', 'standard', 'deep', 'max_detail'
    tokenization_mode='para',    # Options: 'para' (paragraph), 'sent' (sentence)
    model="BAAI/bge-base-en"     # Sentence embedding model (default: "BAAI/bge-base-en")
)
with open("path/to/your/document.txt", "r", encoding="utf-8") as f:
    document = f.read()

chunks = splitter.auto_split(document)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n{chunk.page_content}\n")
```


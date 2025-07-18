import nltk
from nltk.tokenize import sent_tokenize, blankline_tokenize
from sentence_transformers import SentenceTransformer, util
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Union
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure nltk tokenizer data is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

class SemanticSplitter:
    """
    Splits a document into semantically meaningful chunks using Sentence Transformers.
    """

    def __init__(
        self,
        threshold: float = 0.4,
        depth: str = 'light',
        batch_size: int = 64,
        tokenization_mode: str = 'sentence',
        model: Union[str, SentenceTransformer] = "BAAI/bge-base-en"
    ):
        """
        Parameters:
        - threshold: Cosine similarity threshold for semantic grouping (default: 0.4)
        - depth: 'light', 'standard', 'deep', 'max_detail' (controls chunk size/overlap)
        - batch_size: Placeholder for future batching logic
        - tokenization_mode: 'para' or 'sentence'
        - model: HuggingFace model name (str) or a preloaded SentenceTransformer instance
        """
        self.threshold = threshold
        self.depth = depth
        self.batch_size = batch_size
        self.tokenization_mode = tokenization_mode
        self.depth_config = {
            "light": {"N": 16, "overlap_ratio": 0.15},
            "standard": {"N": 13, "overlap_ratio": 0.20},
            "deep": {"N": 10, "overlap_ratio": 0.25},
            "max_detail": {"N": 8, "overlap_ratio": 0.30}
        }

        # Initialize model
        if isinstance(model, str):
            self.model = SentenceTransformer(model)
        else:
            self.model = model

    def _get_chunk_config(self, document_length: int):
        # Case 1: Very long document
        if document_length > 16000:
            chunk_size = 1000
            chunk_overlap = int(0.15 * chunk_size)
            return chunk_size, chunk_overlap

        # Case 2: Very short document
        if document_length < 1000:
            chunk_size = 100
            chunk_overlap = int(0.15 * chunk_size)
            return chunk_size, chunk_overlap

        config = self.depth_config.get(self.depth, self.depth_config["light"])
        chunk_size = document_length / config["N"]
        chunk_overlap = config["overlap_ratio"] * chunk_size
        return int(chunk_size), int(chunk_overlap)

    def _tokenize(self, document: str) -> List[str]:
        return sent_tokenize(document) if self.tokenization_mode == 'sentence' else [
            p.strip() for p in blankline_tokenize(document) if p.strip()]

    def auto_split(self, document: str):
        units = self._tokenize(document)
        if not units:
            return []

        logger.info(f"Tokenized into {len(units)} units using mode: {self.tokenization_mode}")

        embeddings = self.model.encode(units, convert_to_tensor=True, batch_size=self.batch_size)
        cosine_scores = util.pytorch_cos_sim(embeddings, embeddings)
        similarity_matrix = cosine_scores.cpu().numpy()

        used = set()
        groups = []

        for i in range(len(units)):
            if i in used:
                continue
            group = [units[i]]
            used.add(i)
            for j in range(i + 1, len(units)):
                if j not in used and similarity_matrix[i][j] > self.threshold:
                    group.append(units[j])
                    used.add(j)
            groups.append(group)

        semantic_chunks = [" ".join(group) for group in groups]
        chunk_size, chunk_overlap = self._get_chunk_config(len(document))

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return splitter.create_documents(semantic_chunks)
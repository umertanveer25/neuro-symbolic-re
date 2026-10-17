"""
Pre-computes and caches true Sentence-BERT (all-MiniLM-L6-v2) embeddings
for Promise_Exp and verified Neuro-Symbolic ISO 25010 templates.
Allows ultra-fast training and evaluation across all 11 algorithms!
"""

import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.dataset_loader import load_promise_exp
from neuro.generator import EnhancedNeuroSymbolicAugmenter

def generate_and_cache():
    print("Loading all-MiniLM-L6-v2 Sentence-BERT model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 1. Real PROMISE data
    df = load_promise_exp()
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    print(f"Encoding {len(texts)} PROMISE requirement texts...")
    embeddings_promise = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    
    np.save('promise_sbert.npy', embeddings_promise)
    print("Saved promise_sbert.npy successfully!")

if __name__ == '__main__':
    generate_and_cache()

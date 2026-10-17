"""
Pre-encodes verified ISO/IEC 25010 Neuro-Symbolic requirements
into 384-dimensional Sentence-BERT embeddings.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from neuro.generator import EnhancedNeuroSymbolicAugmenter

def encode_iso_corpus():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    augmenter = EnhancedNeuroSymbolicAugmenter(random_seed=42)
    
    classes = ['PE', 'SE', 'US', 'A', 'FT', 'MN', 'SC', 'PO', 'L', 'LF', 'O']
    all_iso_texts = []
    all_iso_labels = []
    
    for c in classes:
        texts, _ = augmenter.generate_verified_batch(c, 35)
        all_iso_texts.extend(texts)
        all_iso_labels.extend([c] * len(texts))
        
    print(f"Encoding {len(all_iso_texts)} verified ISO 25010 requirements...")
    embs = model.encode(all_iso_texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    
    np.save('iso_sbert.npy', embs)
    np.save('iso_labels.npy', np.array(all_iso_labels))
    with open('iso_texts.txt', 'w', encoding='utf-8') as f:
        for t in all_iso_texts:
            f.write(t.replace('\n', ' ') + '\n')
            
    print(f"Saved {len(all_iso_texts)} ISO SBERT embeddings to iso_sbert.npy!")

if __name__ == '__main__':
    encode_iso_corpus()

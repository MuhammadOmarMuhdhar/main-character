from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import json
import os
import logging
from typing import Dict, List, Optional

def encode(posts: List[Dict],
           model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
           batch_size: int = 100,
           device: Optional[str] = None,
           output_path: Optional[str] = None) -> List[Dict]:
    """
    Encode data using SentenceTransformer in batches and append embeddings to the posts.
    """
    model = SentenceTransformer(model_name)

    # Set device
    if device:
        model = model.to(device)
        print(f"Using specified device: {device}")
    else:
        if torch.cuda.is_available():
            model = model.to('cuda')
            print("Using CUDA")
        elif torch.backends.mps.is_available():
            model = model.to('mps')
            print("Using MPS")
        else:
            print("Using CPU")
    
    results = []

    for i in range(0, len(posts), batch_size):
        batch = posts[i:i+batch_size]
        texts = [post['text'] for post in batch if 'text' in post]
        
        # Compute embeddings
        embeddings = model.encode(texts, convert_to_tensor=False, normalize_embeddings=True, show_progress_bar=True)
        
        # Append embeddings back to the posts
        for j, post in enumerate(batch):
            if 'text' in post:
                post['embedding'] = embeddings[j].tolist()  # Convert tensor/array to list for JSON compatibility
                results.append(post)

    # Optionally save to file
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {output_path}")

    return results
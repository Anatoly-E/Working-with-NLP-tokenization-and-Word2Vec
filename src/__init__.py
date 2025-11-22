"""
Модули для лабораторной работы по NLP
"""

from .data_loader import download_nltk_resources, load_gutenberg_corpus, save_processed_data, show_corpus_info
from .preprocessor import TextPreprocessor
from .word2vec_train import Word2VecTrainer
from .visualizer import EmbeddingVisualizer

__all__ = [
    'download_nltk_resources',
    'load_gutenberg_corpus', 
    'save_processed_data',
    'show_corpus_info',
    'TextPreprocessor',
    'Word2VecTrainer',
    'EmbeddingVisualizer'
]
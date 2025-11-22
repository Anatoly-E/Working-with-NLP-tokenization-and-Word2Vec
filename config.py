"""
Конфигурация проекта для лабораторной работы по NLP
"""

# Настройки данных
DATA_CONFIG = {
    'corpus_name': 'gutenberg',  # или 'custom' для своих данных
    'book_name': 'carroll-alice.txt',  # для gutenberg
    'custom_file_path': 'data/raw/custom_text.txt'  # для своих данных
}

# Настройки предобработки
PREPROCESSING_CONFIG = {
    'language': 'english',  # 'english' или 'russian'
    'remove_stopwords': True,
    'do_lemmatization': True,
    'min_word_length': 2
}

# Настройки Word2Vec
WORD2VEC_CONFIG = {
    'vector_size': 100,
    'window': 5,
    'min_count': 3,
    'workers': 4,
    'sg': 1,  # 1 для skip-gram, 0 для CBOW
    'epochs': 10
}

# Настройки визуализации
VISUALIZATION_CONFIG = {
    'words_to_plot': ['alice', 'rabbit', 'queen', 'king', 'cat', 'hatter', 
                     'door', 'garden', 'time', 'mouse', 'head', 'tea'],
    'plot_size': (12, 8),
    'random_state': 42 # Это случайное число везде неслучайно ;)
}
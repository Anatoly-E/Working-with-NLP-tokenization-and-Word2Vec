"""
Конфигурация проекта для лабораторной работы по NLP
"""

# Настройки для разных языков
LANGUAGE_CONFIGS = {
    'english': {
        'corpus_name': 'gutenberg',
        'book_name': 'carroll-alice.txt',
        'preprocessing': {
            'language': 'english',
            'remove_stopwords': True,
            'do_lemmatization': True,
            'min_word_length': 2
        },
        'word2vec': {
            'vector_size': 100,
            'window': 5,
            'min_count': 2,
            'workers': 4,
            'sg': 1,
            'epochs': 20
        },
        'visualization': {
            'words_to_plot': [
                'alice', 'rabbit', 'queen', 'king', 'card', 'door', 'garden',
                'time', 'mouse', 'head', 'tea', 'hatter', 'cat', 'dog',
                'mushroom', 'book', 'soup', 'nose', 'watch', 'hat'
            ],
            'plot_size': (12, 8),
            'random_state': 42
        },
        'test_words': ['alice', 'rabbit', 'queen', 'king', 'time', 'tea']
    },
    'russian': {
        'corpus_name': 'russian_alice',
        'book_name': 'russian-text.txt',
        'preprocessing': {
            'language': 'russian',
            'remove_stopwords': True,
            'do_lemmatization': True,
            'min_word_length': 2
        },
        'word2vec': {
            'vector_size': 100,
            'window': 5,
            'min_count': 2,
            'workers': 4,
            'sg': 1,
            'epochs': 20
        },
        'visualization': {
            'words_to_plot': [
                'алиса', 'кролик', 'королева', 'король', 'карта', 'дверь', 'сад',
                'время', 'мышь', 'голова', 'чай', 'шляпа', 'кот', 'собака',
                'гриб', 'книга', 'суп', 'нос', 'часы', 'заяц', 'черепаха',
                'шахматы', 'приключение', 'сон', 'сестра', 'река', 'берег'
            ],
            'plot_size': (12, 8),
            'random_state': 42
        },
        'test_words': ['алиса', 'кролик', 'королева', 'король', 'время', 'чай', 
            'шляпа', 'заяц', 'сон', 'приключение']
    }
}

# Текущий язык (будет выбран пользователем)
CURRENT_LANGUAGE = 'russian'  # По умолчанию

# Пути к файлам
DATA_PATHS = {
    'russian_file_path': 'data/raw/russian_text.txt',
    'custom_file_path': 'data/raw/custom_text.txt'
}
"""
Загрузка данных для лабораторной работы по NLP
"""

import nltk
from nltk.corpus import gutenberg
import os

def download_nltk_resources():
    """Скачивает необходимые ресурсы NLTK"""
    print("📥 Скачиваем ресурсы NLTK...")
    
    # Основные ресурсы
    resources = [
        'punkt',           # Токенизатор
        'punkt_tab',       # Дополнительные таблицы для токенизатора
        'stopwords',       # Стоп-слова
        'wordnet',         # Лемматизатор
        'omw-1.4',         # Open Multilingual WordNet
        'gutenberg'        # Корпус текстов
    ]
    
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
            print(f"   ✅ {resource}")
        except Exception as e:
            print(f"   ⚠️ Ошибка с {resource}: {e}")
    
    print("✅ Все ресурсы NLTK загружены!")

def load_gutenberg_corpus(book_name='carroll-alice.txt'):
    """Загружает корпус из библиотеки Gutenberg"""
    try:
        print(f"📚 Загружаем книгу: {book_name}")
        text = gutenberg.raw(book_name)
        print(f"✅ Успешно загружено!")
        print(f"📊 Размер текста: {len(text):,} символов")
        print(f"📖 Первые 300 символов:\n{text[:300]}...")
        return text
    except Exception as e:
        print(f"❌ Ошибка загрузки текста: {e}")
        return None

def load_custom_corpus(file_path):
    """Загружает пользовательский корпус из файла"""
    try:
        print(f"📁 Загружаем файл: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        print(f"✅ Успешно загружено!")
        print(f"📊 Размер текста: {len(text):,} символов")
        return text
    except Exception as e:
        print(f"❌ Ошибка загрузки файла: {e}")
        return None

def save_processed_data(tokens, filename='processed_tokens.txt'):
    """Сохраняет обработанные токены"""
    os.makedirs('data/processed', exist_ok=True)
    filepath = os.path.join('data/processed', filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for token in tokens:
            f.write(token + '\n')
    
    print(f"💾 Обработанные данные сохранены: {filepath}")
    print(f"📝 Сохранено токенов: {len(tokens):,}")

def show_corpus_info():
    """Показывает доступные корпуса в Gutenberg"""
    try:
        file_ids = gutenberg.fileids()
        print("\n📚 Доступные книги в Gutenberg:")
        for i, file_id in enumerate(file_ids[:10]):  # покажем первые 10
            print(f"   {i+1}. {file_id}")
        if len(file_ids) > 10:
            print(f"   ... и еще {len(file_ids) - 10} книг")
        return file_ids
    except Exception as e:
        print(f"❌ Не удалось получить список книг: {e}")
        return []
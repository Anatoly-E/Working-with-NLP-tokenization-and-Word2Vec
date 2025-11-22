"""
Главный скрипт для лабораторной работы по NLP
Токенизация и Word2Vec
"""

import sys
import os

# Добавляем папку src в путь для импорта
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Импортируем только то, что точно есть в модулях
from config import LANGUAGE_CONFIGS, DATA_PATHS, CURRENT_LANGUAGE
from src.data_loader import download_nltk_resources, load_gutenberg_corpus, load_russian_corpus, load_custom_corpus, save_processed_data
from src.preprocessor import TextPreprocessor
from src.word2vec_train import Word2VecTrainer
from src.visualizer import EmbeddingVisualizer
    
def select_language():
    """Позволяет пользователю выбрать язык"""
    print("🌍 ВЫБЕРИТЕ ЯЗЫК ДЛЯ АНАЛИЗА:")
    print("   1. Английский (Alice in Wonderland)")
    print("   2. Русский (Алиса в Стране чудес)")
    
    while True:
        choice = input("\n🎯 Введите 1 или 2: ").strip()
        if choice == '1':
            return 'english'
        elif choice == '2':
            return 'russian'
        else:
            print("❌ Пожалуйста, введите 1 или 2")

def get_language_config(language):
    """Возвращает конфигурацию для выбранного языка"""
    return LANGUAGE_CONFIGS[language]

def main():
    print("🚀 Запуск лабораторной работы: NLP с Word2Vec")
    print("=" * 50)
    
    # Выбор языка
    language = select_language()
    config = get_language_config(language)
    
    print(f"\n🌍 Выбран язык: {language.upper()}")
    print("=" * 50)
    
    # Шаг 1: Подготовка и загрузка данных
    print("\n📥 ШАГ 1: Загрузка данных...")
    download_nltk_resources()
    
    # Загружаем данные в зависимости от языка
    if language == 'english':
        text = load_gutenberg_corpus(config['book_name'])
        corpus_type = "Gutenberg (английский)"
    else:  # russian
        text = load_russian_corpus(DATA_PATHS['russian_file_path'])
        corpus_type = "Русский текст"
    
    if not text:
        print("❌ Не удалось загрузить данные!")
        return
    
    print(f"📚 Используется: {corpus_type}")
    
    # Шаг 2: Предобработка текста
    print("\n🛠️ ШАГ 2: Предобработка текста...")
    preprocessor = TextPreprocessor(language=config['preprocessing']['language'])
    processed_tokens = preprocessor.preprocess_text(text)
    
    # Диагностика наличия ключевых слов
    print(f"\n🔍 Проверяем наличие ключевых слов ({language}):")
    from collections import Counter
    word_freq = Counter(processed_tokens)
    
    test_words = config['test_words']
    found_words = []
    
    for word in test_words:
        if word in word_freq:
            print(f"   ✅ '{word}': найдено ({word_freq[word]} раз)")
            found_words.append(word)
        else:
            print(f"   ❌ '{word}': не найдено")
    
    if len(found_words) < 3:
        print(f"⚠️ Мало ключевых слов найдено. Используем самые частые слова.")
        found_words = [word for word, count in word_freq.most_common(10)]
    
    # Сохраняем обработанные данные
    filename = f"processed_tokens_{language}.txt"
    save_processed_data(processed_tokens, filename)
    
    # Шаг 3: Обучение Word2Vec
    print("\n🎯 ШАГ 3: Обучение Word2Vec модели...")
    trainer = Word2VecTrainer(config['word2vec'])
    model = trainer.train_model(processed_tokens)
    
    # Исследуем модель
    print(f"\n🔍 Исследуем модель ({language}):")
    trainer.explore_model(found_words)
    
    # Сохраняем модель
    model_path = f'models/word2vec_model_{language}'
    trainer.save_model(model_path)
    
    # Шаг 4: Визуализация
    print("\n🎨 ШАГ 4: Визуализация эмбеддингов...")
    visualizer = EmbeddingVisualizer(config['visualization'], language=language)
    success = visualizer.plot_embeddings(model)
    
    if not success:
        print("⚠️ Не удалось создать визуализацию")
    
    print("\n" + "=" * 50)
    print(f"✅ Лабораторная работа завершена успешно! (Язык: {language})")
    print("📁 Результаты сохранены в папках:")
    print(f"   - models/word2vec_model_{language}/word2vec.model")
    print(f"   - results/plots/word_embeddings.png")
    print(f"   - data/processed/processed_tokens_{language}.txt")

if __name__ == "__main__":
    main()
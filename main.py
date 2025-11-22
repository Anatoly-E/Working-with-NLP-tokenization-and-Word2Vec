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

try:
    # Импортируем только то, что точно есть в модулях
    from src.data_loader import download_nltk_resources, load_gutenberg_corpus, load_custom_corpus, save_processed_data
    from src.preprocessor import TextPreprocessor
    from src.word2vec_train import Word2VecTrainer
    from src.visualizer import EmbeddingVisualizer
    
    # Пробуем импортировать show_corpus_info, но если нет - создадим свою версию
    try:
        from src.data_loader import show_corpus_info
    except ImportError:
        print("Функция show_corpus_info не найдена, создаем свою...")
        def show_corpus_info():
            """Простая версия показа информации о корпусе"""
            try:
                from nltk.corpus import gutenberg
                file_ids = gutenberg.fileids()
                print("\nДоступные книги в Gutenberg:")
                for i, file_id in enumerate(file_ids[:5]):
                    print(f"   {i+1}. {file_id}")
                return file_ids
            except:
                print("Не удалось получить список книг")
                return []
    
    # Импортируем конфиг
    from config import DATA_CONFIG, PREPROCESSING_CONFIG, WORD2VEC_CONFIG, VISUALIZATION_CONFIG
    
    print("Все модули успешно импортированы!")
    
except ImportError as e:
    print(f"Критическая ошибка импорта: {e}")
    print("\nПроверьте содержимое файлов в папке src/")
    sys.exit(1)

def main():
    print("\nЗапуск лабораторной работы: NLP с Word2Vec")
    print("=" * 50)
    
    # Шаг 1: Подготовка и загрузка данных
    print("\nШАГ 1: Загрузка данных...")
    download_nltk_resources()
    
    # Покажем доступные книги
    available_books = show_corpus_info()
    
    if DATA_CONFIG['corpus_name'] == 'gutenberg':
        text = load_gutenberg_corpus(DATA_CONFIG['book_name'])
    else:
        text = load_custom_corpus(DATA_CONFIG['custom_file_path'])
    
    if not text:
        print("Не удалось загрузить данные!")
        return
    
    # Шаг 2: Предобработка текста
    print("\nШАГ 2: Предобработка текста...")
    preprocessor = TextPreprocessor(language=PREPROCESSING_CONFIG['language'])
    processed_tokens = preprocessor.preprocess_text(text)
    
    # Сохраняем обработанные данные
    save_processed_data(processed_tokens)
    
    # Показываем пример токенов
    sample_tokens = preprocessor.get_sample_tokens(processed_tokens, 15)
    print(f"Пример токенов: {sample_tokens}")
    
    # Шаг 3: Обучение Word2Vec
    print("\nШАГ 3: Обучение Word2Vec модели...")
    trainer = Word2VecTrainer(WORD2VEC_CONFIG)
    model = trainer.train_model(processed_tokens)
    
    # Проверяем, что модель успешно обучена
    if model is None:
        print("Не удалось обучить модель Word2Vec!")
        print("Возможные причины:")
        print(" - Слишком мало данных после предобработки")
        print(" - Параметр min_count слишком высокий")
        print(" - Проблемы с входными данными")
        return
    
    # Исследуем модель
    trainer.explore_model(['alice', 'rabbit', 'queen', 'king'])
    
    # Сохраняем модель
    success = trainer.save_model()
    if not success:
        print("Не удалось сохранить модель, но продолжаем...") 
           
    # Шаг 4: Визуализация
    print("\nШАГ 4: Визуализация эмбеддингов...")
    visualizer = EmbeddingVisualizer(VISUALIZATION_CONFIG)
    visualizer.plot_embeddings(model)
    
    print("\n" + "=" * 50)
    print("Лабораторная работа завершена успешно!")
    print("Результаты сохранены в папках:")
    print(" - models/word2vec_model/word2vec.model")
    print(" - results/plots/word_embeddings.png")
    print(" - data/processed/processed_tokens.txt")

if __name__ == "__main__":
    main()
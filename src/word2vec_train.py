"""
Обучение модели лабораторной работы по NLP
"""

from gensim.models import Word2Vec
import os

class Word2VecTrainer:
    def __init__(self, config):
        self.config = config
        self.model = None
        
    def train_model(self, tokens):
        """Обучает модель Word2Vec"""
        print("🎯 Начинаем обучение Word2Vec...")
        
        # Проверяем входные данные
        if not tokens or len(tokens) == 0:
            print("❌ Ошибка: передан пустой список токенов!")
            return None
        
        print(f"📊 Обучаем на {len(tokens):,} токенах...")
        
        try:
            # Подготавливаем данные в формате для Word2Vec
            sentences = [tokens]  # Используем весь текст как одно "предложение"
            
            self.model = Word2Vec(
                sentences=sentences,
                vector_size=self.config['vector_size'],
                window=self.config['window'],
                min_count=self.config['min_count'],
                workers=self.config['workers'],
                sg=self.config['sg'],
                epochs=self.config['epochs']
            )
            
            print("✅ Модель Word2Vec успешно обучена!")
            print(f"📚 Размер словаря: {len(self.model.wv.key_to_index):,} слов")
            
            # Покажем несколько слов из словаря
            vocab_words = list(self.model.wv.key_to_index.keys())[:10]
            print(f"📋 Примеры слов в словаре: {vocab_words}")
            
            return self.model
            
        except Exception as e:
            print(f"❌ Ошибка при обучении модели: {e}")
            return None
    
    def explore_model(self, test_words=None):
        """Исследует обученную модель"""
        if not self.model:
            print("❌ Модель не обучена! Нечего исследовать.")
            return
        
        if test_words is None:
            test_words = ['alice', 'rabbit', 'queen']
        
        print("\n🔍 Исследуем модель:")
        
        for word in test_words:
            if word in self.model.wv.key_to_index:
                similar = self.model.wv.most_similar(word, topn=3)
                print(f"📌 Слова похожие на '{word}': {similar}")
            else:
                print(f"⚠️ Слово '{word}' не найдено в словаре")
        
        # Проверяем сходство между словами
        if all(word in self.model.wv.key_to_index for word in ['queen', 'king']):
            similarity = self.model.wv.similarity('queen', 'king')
            print(f"📊 Сходство 'queen' и 'king': {similarity:.3f}")
    
    def save_model(self, model_path='models/word2vec_model'):
        """Сохраняет обученную модель"""
        if not self.model:
            print("❌ Нечего сохранять - модель не обучена!")
            return False
            
        try:
            os.makedirs(model_path, exist_ok=True)
            model_file = f"{model_path}/word2vec.model"
            self.model.save(model_file)
            print(f"💾 Модель сохранена: {model_file}")
            return True
        except Exception as e:
            print(f"❌ Ошибка при сохранении модели: {e}")
            return False
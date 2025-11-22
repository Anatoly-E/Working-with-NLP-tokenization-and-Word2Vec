"""
Визуализация результатов лабораторной работы по NLP
"""

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import os

class EmbeddingVisualizer:
    def __init__(self, config, language="russian"):
        self.config = config
        self.language = language
        plt.style.use('default')
    
    def plot_embeddings(self, model, words_to_plot=None):
        """Визуализирует эмбеддинги с помощью PCA"""
        
        # Проверяем модель
        if model is None:
            print("❌ Модель не передана для визуализации!")
            return False
            
        if not hasattr(model, 'wv') or not hasattr(model.wv, 'key_to_index'):
            print("❌ Переданная модель не является корректной Word2Vec моделью!")
            return False
        
        if words_to_plot is None:
            words_to_plot = self.config['words_to_plot']
        
        # Фильтруем слова, которые есть в модели
        available_words = [word for word in words_to_plot if word in model.wv.key_to_index]
        
        print(f"🔍 Найдено слов для визуализации: {len(available_words)} из {len(words_to_plot)}")
        
        if len(available_words) < 3:
            print(f"❌ Недостаточно слов для визуализации. Доступные слова: {available_words}")
            print(f"📊 Все слова в модели: {list(model.wv.key_to_index.keys())[:20]}...")
            return False
        
        print(f"🎨 Визуализируем {len(available_words)} слов...")
        
        try:
            # Получаем векторы
            vectors = [model.wv[word] for word in available_words]
            vectors_array = np.array(vectors)
            
            # Применяем PCA
            pca = PCA(n_components=2, random_state=self.config['random_state'])
            vectors_2d = pca.fit_transform(vectors_array)
            
            # Создаем визуализацию
            self._create_plot(vectors_2d, available_words)
            
            # Сохраняем график
            self._save_plot()
            return True
            
        except Exception as e:
            print(f"❌ Ошибка при визуализации: {e}")
            return False
    
    def _create_plot(self, vectors_2d, words):
        """Создает график с эмбеддингами"""
        plt.figure(figsize=self.config['plot_size'])
        
        # Scatter plot
        scatter = plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], 
                            s=150, alpha=0.7, c=range(len(words)), 
                            cmap='viridis')
        
        # Добавляем подписи
        for i, word in enumerate(words):
            plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]), 
                        fontsize=11, alpha=0.9,
                        bbox=dict(boxstyle="round,pad=0.3", 
                                facecolor="lightblue", alpha=0.7),
                        ha='center', va='center')

        plt.title('Визуализация векторных представлений слов Word2Vec при помощи PCA\n', fontsize=14, pad=20)
        plt.xlabel('Главная компонента 1')
        plt.ylabel('Главная компонента 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Добавляем colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Word Index', rotation=270, labelpad=15)
    
    def _save_plot(self):
        # Добавляем язык в название файла
        language_name = "russian_text" if self.language == 'russian' else "gothenberg"
        
        """Сохраняет график в папку results"""
        os.makedirs('results/plots', exist_ok=True)
        plot_path = f'results/plots/word_embeddings_{language_name}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"💾 График сохранен: {plot_path}")
        plt.show()
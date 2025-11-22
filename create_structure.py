"""
Структура файлов и папок для лабораторной работы по NLP
"""

import os

def create_project_structure():
    folders = [
        'src',
        'data/raw',
        'data/processed', 
        'models/word2vec_model',
        'results/embeddings',
        'results/plots'
    ]
    
    files = [
        'src/__init__.py',
        'src/data_loader.py',
        'src/preprocessor.py', 
        'src/word2vec_train.py',
        'src/visualizer.py',
        'config.py',
        'main.py'
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"📁 Создана папка: {folder}")
    
    for file in files:
        with open(file, 'w', encoding='utf-8') as f:
            if file == 'src/__init__.py':
                f.write('''"""
Модули для лабораторной работы по NLP
"""''')
        print(f"📄 Создан файл: {file}")

if __name__ == "__main__":
    create_project_structure()
    print("✅ Структура проекта создана!")
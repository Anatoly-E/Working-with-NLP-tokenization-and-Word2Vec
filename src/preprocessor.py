"""
Предобработка текста лабораторной работы по NLP
"""

import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer  # Для русского стемминга
import pymorphy3  # Для русской лемматизации

class TextPreprocessor:
    def __init__(self, language='russian'):
        self.language = language

        if language == 'russian':
            # Русские стоп-слова
            self.stop_words = set(stopwords.words('russian'))
            # Добавляем дополнительные русские стоп-слова
            russian_stopwords_extended = [
                'это', 'как', 'так', 'и', 'в', 'над', 'к', 'до', 'не', 'на', 'но', 'за', 
                'то', 'с', 'ли', 'а', 'во', 'от', 'со', 'для', 'о', 'же', 'ну', 'вы', 
                'бы', 'что', 'кто', 'он', 'она'
            ]
            self.stop_words.update(russian_stopwords_extended)
            
            # Русская пунктуация
            self.punctuation = set(string.punctuation + '«»—…')
            
            # Инициализируем морфологический анализатор для русского
            try:
                self.morph = pymorphy3.MorphAnalyzer()
                print("✅ PyMorphy3 загружен для русской лемматизации")
            except ImportError:
                print("❌ PyMorphy3 не установлен. Установите: pip install pymorphy3")
                self.morph = None
                
            self.stemmer = SnowballStemmer('russian')
            
        else:  # english
            self.stop_words = set(stopwords.words('english'))
            self.punctuation = set(string.punctuation)
            self.lemmatizer = WordNetLemmatizer()
            self.morph = None
            self.stemmer = None

    def preprocess_text(self, text):
        """Основная функция предобработки текста"""
        print("🔄 Начинаем предобработку текста...")
        
        # Токенизация
        tokens = word_tokenize(text.lower())
        print(f"📝 Токенизировано слов: {len(tokens):,}")
        
        # Очистка
        cleaned_tokens = self._clean_tokens(tokens)
        
        # Лемматизация/стемминг
        if self.language == 'russian' and self.morph:
            cleaned_tokens = self._lemmatize_russian(cleaned_tokens)
        elif self.language == 'english':
            cleaned_tokens = self._lemmatize_english(cleaned_tokens)
        
        print(f"✨ После очистки осталось: {len(cleaned_tokens):,} токенов")
        return cleaned_tokens
    
    def _clean_tokens(self, tokens):
        """Очистка токенов от стоп-слов и пунктуации"""
        cleaned = []
        for token in tokens:
            if (token not in self.stop_words and 
                token not in self.punctuation and
                token.isalpha() and 
                len(token) > 2):
                cleaned.append(token)
        return cleaned
    
    def _lemmatize_russian(self, tokens):
        """Лемматизация для русского языка с помощью pymorphy3"""
        print("🔤 Лемматизируем русские слова...")
        lemmatized = []
        for token in tokens:
            try:
                # Получаем нормальную форму слова
                parsed = self.morph.parse(token)[0]
                lemma = parsed.normal_form
                lemmatized.append(lemma)
            except Exception as e:
                # В случае ошибки оставляем оригинальное слово
                lemmatized.append(token)
        return lemmatized
    
    def _lemmatize_english(self, tokens):
        """Лемматизация для английского языка"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def _lemmatize_tokens(self, tokens):
        """Лемматизация токенов"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def get_sample_tokens(self, tokens, n=20):
        """Возвращает пример токенов для проверки"""
        return tokens[:n]
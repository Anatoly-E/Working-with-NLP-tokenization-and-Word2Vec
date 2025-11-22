"""
Предобработка текста лабораторной работы по NLP
"""

import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class TextPreprocessor:
    def __init__(self, language='english'):
        self.language = language
        self.stop_words = set(stopwords.words(language))
        self.punctuation = set(string.punctuation)
        self.lemmatizer = WordNetLemmatizer()
        
    def preprocess_text(self, text):
        """Основная функция предобработки текста"""
        print("🔄 Начинаем предобработку текста...")
        
        # Токенизация
        tokens = word_tokenize(text.lower())
        print(f"📝 Токенизировано слов: {len(tokens):,}")
        
        # Очистка
        cleaned_tokens = self._clean_tokens(tokens)
        
        # Лемматизация
        cleaned_tokens = self._lemmatize_tokens(cleaned_tokens)
        
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
    
    def _lemmatize_tokens(self, tokens):
        """Лемматизация токенов"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def get_sample_tokens(self, tokens, n=20):
        """Возвращает пример токенов для проверки"""
        return tokens[:n]
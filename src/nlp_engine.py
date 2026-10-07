import json
import math
import os
import random
import re
from collections import Counter

# Try importing scikit-learn; fall back to pure-Python TF-IDF if not yet installed
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class DepartmentChatbot:
    """
    A foundational NLP Chatbot designed for 3rd Year AI & DS Department mini-projects.
    Uses TF-IDF (Term Frequency - Inverse Document Frequency) and Cosine Similarity
    for pattern matching and intent recognition.
    Supports both scikit-learn and pure standard-library fallback.
    """

    def __init__(self, data_path: str = None):
        if data_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.join(base_dir, "data", "intents.json")

        self.data_path = data_path
        self.intents = []
        self.patterns = []
        self.pattern_to_tag = []
        self.use_sklearn = SKLEARN_AVAILABLE

        # Sklearn components
        if self.use_sklearn:
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
            self.tfidf_matrix = None
        else:
            # Pure Python TF-IDF components
            self.vocabulary = {}
            self.idf_dict = {}
            self.pattern_vectors = []

        self.load_intents()
        self.train_engine()

    def clean_text(self, text: str) -> str:
        """Standardizes input string by lowercasing and removing punctuation."""
        text = text.lower().strip()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        return text

    def load_intents(self):
        """Loads intent patterns and responses from the JSON knowledge base."""
        with open(self.data_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            self.intents = data.get("intents", [])

        for intent in self.intents:
            tag = intent["tag"]
            for pattern in intent.get("patterns", []):
                cleaned = self.clean_text(pattern)
                self.patterns.append(cleaned)
                self.pattern_to_tag.append(tag)

    def _tokenize(self, text: str) -> list:
        return [w for w in text.split() if len(w) > 1]

    def train_engine(self):
        """Builds the TF-IDF representation of all known query patterns."""
        if not self.patterns:
            return

        if self.use_sklearn:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.patterns)
        else:
            # Tokenize all patterns
            tokenized_docs = [self._tokenize(p) for p in self.patterns]
            total_docs = len(tokenized_docs)

            # Build vocabulary
            vocab = set()
            for doc in tokenized_docs:
                vocab.update(doc)
            self.vocabulary = {word: idx for idx, word in enumerate(sorted(vocab))}

            # Calculate IDF: log((N + 1) / (df + 1)) + 1
            for word in self.vocabulary:
                df = sum(1 for doc in tokenized_docs if word in doc)
                self.idf_dict[word] = math.log((total_docs + 1) / (df + 1)) + 1.0

            # Vectorize each pattern
            self.pattern_vectors = [self._pure_vectorize(doc) for doc in tokenized_docs]

    def _pure_vectorize(self, tokens: list) -> list:
        """Transforms a token list into a normalized TF-IDF vector."""
        vec = [0.0] * len(self.vocabulary)
        if not tokens or not self.vocabulary:
            return vec

        counts = Counter(tokens)
        total = len(tokens)

        for word, count in counts.items():
            if word in self.vocabulary:
                idx = self.vocabulary[word]
                tf = count / total
                idf = self.idf_dict.get(word, 1.0)
                vec[idx] = tf * idf

        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def _pure_cosine_similarity(self, vec1: list, vec2: list) -> float:
        """Calculates dot product between two normalized vectors."""
        return sum(a * b for a, b in zip(vec1, vec2))

    def get_response(self, user_query: str, confidence_threshold: float = 0.20) -> dict:
        """
        Processes a user query and returns the best matching response.

        Returns:
            dict containing:
                - response: string response text
                - tag: matched intent tag
                - confidence: cosine similarity score (0.0 to 1.0)
        """
        cleaned_query = self.clean_text(user_query)
        if not cleaned_query:
            return {
                "response": "Please type a question or topic so I can assist you.",
                "tag": "empty",
                "confidence": 0.0
            }

        if self.use_sklearn:
            query_vector = self.vectorizer.transform([cleaned_query])
            similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
            best_match_idx = int(similarities.argmax())
            max_score = float(similarities[best_match_idx])
        else:
            query_tokens = self._tokenize(cleaned_query)
            query_vector = self._pure_vectorize(query_tokens)
            similarities = [
                self._pure_cosine_similarity(query_vector, pat_vec)
                for pat_vec in self.pattern_vectors
            ]
            if similarities:
                max_score = max(similarities)
                best_match_idx = similarities.index(max_score)
            else:
                max_score = 0.0
                best_match_idx = 0

        if max_score < confidence_threshold:
            return {
                "response": (
                    "I am not sure about that yet. You can ask about:\n"
                    "- 3rd year syllabus & subjects\n"
                    "- Labs & practicals\n"
                    "- Mini-project guidelines\n"
                    "- Department HOD & faculty\n"
                    "- Attendance criteria"
                ),
                "tag": "unknown",
                "confidence": round(max_score, 3)
            }

        matched_tag = self.pattern_to_tag[best_match_idx]

        # Select a response from the matched intent
        for intent in self.intents:
            if intent["tag"] == matched_tag:
                selected_reply = random.choice(intent["responses"])
                return {
                    "response": selected_reply,
                    "tag": matched_tag,
                    "confidence": round(max_score, 3)
                }

        return {
            "response": "I encountered an error retrieving the answer.",
            "tag": "error",
            "confidence": 0.0
        }


if __name__ == "__main__":
    bot = DepartmentChatbot()
    print("AI & DS Bot initialized. Type 'quit' to exit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["quit", "exit"]:
            break
        result = bot.get_response(user_input)
        print(f"Bot ({result['tag']}, confidence: {result['confidence']}):\n{result['response']}")

import math
import re
from collections import defaultdict

class NaiveBayesSpamClassifier:
    def __init__(self):
        self.spam_word_counts = defaultdict(int)
        self.ham_word_counts = defaultdict(int)
        self.spam_messages = 0
        self.ham_messages = 0
        self.spam_total_words = 0
        self.ham_total_words = 0
        self.vocab = set()

    def tokenize(self, text):
        return re.findall(r'\b[a-z]{2,}\b', text.lower()) # I don't know how this regex works
    # Build it using standard regexer from the web

    def train(self, messages):

        for text, label in messages:
            words = self.tokenize(text)
            if label == 'spam':
                self.spam_messages += 1
                for w in words:
                    self.spam_word_counts[w] += 1
                    self.spam_total_words += 1
                    self.vocab.add(w)
            else:
                self.ham_messages += 1
                for w in words:
                    self.ham_word_counts[w] += 1
                    self.ham_total_words += 1
                    self.vocab.add(w)

    def predict(self, text):
        words = self.tokenize(text)
        vocab_size = len(self.vocab)

        # log priors
        log_spam = math.log(self.spam_messages / (self.spam_messages + self.ham_messages))
        log_ham  = math.log(self.ham_messages  / (self.spam_messages + self.ham_messages))

        for w in words:
            spam_prob = (self.spam_word_counts[w] + 1) / (self.spam_total_words + vocab_size)
            ham_prob  = (self.ham_word_counts[w]  + 1) / (self.ham_total_words  + vocab_size)

            log_spam += math.log(spam_prob)
            log_ham  += math.log(ham_prob)

        return 'spam' if log_spam > log_ham else 'ham'


# ------------------ Example usage ------------------

training_data = [
    ("Win money now", "spam"),
    ("Cheap loans available", "spam"),
    ("Limited offer claim prize", "spam"),
    ("Meeting at 10 am", "ham"),
    ("Let's have lunch tomorrow", "ham"),
    ("Project deadline extended", "ham")
]

classifier = NaiveBayesSpamClassifier()
classifier.train(training_data)

tests = [
    "Win a cheap prize now",
    "Are we meeting tomorrow"
]

for msg in tests:
    print(f"{msg} -> {classifier.predict(msg)}")
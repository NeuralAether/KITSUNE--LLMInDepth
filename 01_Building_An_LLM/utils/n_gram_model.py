import random 
from collections import defaultdict, Counter, deque
import pandas as pd 

class NGramModel:
    def __init__(self, dataset, n): 
        self.dataset = dataset
        self.n = n
        self.tokens_generator = self.__tokenize()
        self.ngrams_generator = self.__generate_ngrams()
        self.ngram_counts = self.__get_ngram_counts()
        self.ngram_model = self.__build_ngram_model()

    def __tokenize(self):
        for sentence in self.dataset:
            tokens = sentence.split(" ")
            for token in tokens:
                yield token

    def __generate_ngrams(self):
        # Yield n-grams as tuples of tokens
        token_queue = deque(maxlen=self.n)
        for token in self.tokens_generator:
            token_queue.append(token)
            if len(token_queue) == self.n:
                yield tuple(token_queue)

    def __get_ngram_counts(self):
        ngram_counts = defaultdict(Counter)
        for ngram in self.ngrams_generator:
            context = ngram[:-1]  # All tokens except the last one
            next_token = ngram[-1]  # The last token is the next token to predict
            ngram_counts[" ".join(context)][next_token] += 1
        return dict(ngram_counts)

    def __build_ngram_model(self):
        ngram_model = {}
        for context, next_tokens in self.ngram_counts.items():
            total_count = sum(next_tokens.values())
            ngram_model[context] = {token: count / total_count for token, count in next_tokens.items()}
        return ngram_model

    def predict_next_token(self, context):
        if context not in self.ngram_model:
            return "Context not found in model."  # No predictions available for this context
        next_token_probs = self.ngram_model[context]
        predicted_token = random.choices(list(next_token_probs.keys()), weights=next_token_probs.values(), k=1)[0]
        return predicted_token
    
    def generate_text(self, start_context, max_new_tokens):
        context = start_context.split(" ")[-self.n+1:]  # Get the last n-1 tokens from the start context
        context = " ".join(context)
        generated_tokens = []
        for _ in range(max_new_tokens):
            next_token = self.predict_next_token(context)
            if next_token == "Context not found in model.":
                generated_tokens.append("[END PREMATURELY]")
                break 
            generated_tokens.append(next_token)
            context_words = context.split(" ")
            context_words.append(next_token)
            context = " ".join(context_words[1:])
        return start_context + " " + " ".join(generated_tokens)  # Update context to the last n-1 tokens

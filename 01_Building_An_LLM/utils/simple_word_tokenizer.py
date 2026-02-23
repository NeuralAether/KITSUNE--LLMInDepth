import re

class SimpleWordTokenizer:
    UNKOWN_TOKEN = "<UNK>"
    PAD_TOKEN = "<PAD>" 
    def __init__(self) -> None : 
        # Stating the variables 
        self.vocabulary = None 
        self.vocabulary_size = None
        self.token_to_id = {}
        self.id_to_token = {}
        self.pad_token_id = None
        self.unknown_token_id = None

    def adapt(self, corpus: str | list[str] , vocabulary: list[str] | None = None) -> None : 
        # 1) Setting the vocabulary
        if vocabulary is None: 
            # Will have to build the vocabulary from the corpus
            if isinstance(corpus, str): 
                corpus = [corpus]
            # Tokenization 
            tokens = []
            for text in corpus: 
                tokens.extend(text.split())
            # Building the vocabulary
            vocabulary = sorted(list(set(tokens)))
            # Adding the special tokens to the vocabulary
            self.vocabulary = [self.PAD_TOKEN] + vocabulary + [self.UNKOWN_TOKEN]
        else : 
            self.vocabulary = vocabulary
        # 2) Sizing the vocabulary
        self.vocabulary_size = len(self.vocabulary)
        # 3) Building the token to id and id to token mappings
        for index, token in enumerate(self.vocabulary): 
            self.token_to_id[token] = index
            self.id_to_token[index] = token
        # 4) Setting the pad token id and unknown token id
        self.pad_token_id = self.token_to_id[self.PAD_TOKEN]
        self.unknown_token_id = self.token_to_id[self.UNKOWN_TOKEN]

    def encode(self, text: str) -> list[int] : 
        # Converting the text to tokens 
        indices = []
        unk_index = self.token_to_id[self.UNKOWN_TOKEN]
        for token in re.split(" +", text):
            token_id = self.token_to_id.get(token, unk_index)
            indices.append(token_id)
        return indices
    
    def decode(self, indices: int | list[int]) -> str : 
        # Converting to list if indices is a single integer
        if isinstance(indices, int):
            indices = [indices]
        
        # Mapping the indices to tokens
        tokens = []
        for index in indices: 
            token = self.id_to_token.get(index, self.UNKOWN_TOKEN)
            tokens.append(token)
        return " ".join(tokens)
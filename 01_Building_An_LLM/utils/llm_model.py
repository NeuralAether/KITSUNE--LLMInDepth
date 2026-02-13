import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLMModel:
    def __init__(self, model_path="__assets/gemma-3-270m-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32).to("mps")

    def generate_text(self, prompt, **kwargs):
        inputs = self.tokenizer(prompt, return_tensors="pt").to("mps")
        outputs = self.model.generate(**inputs, **kwargs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def return_probabilities(self, prompt, k=10):
        inputs = self.tokenizer(prompt, return_tensors="pt").to("mps")
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits[:, -1, :]  # logits for the next token
        probabilities = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probabilities, k, dim=-1)
        top_tokens = [self.tokenizer.decode(idx) for idx in top_indices[0]]
        return list(zip(top_tokens, top_probs[0].cpu().tolist()))
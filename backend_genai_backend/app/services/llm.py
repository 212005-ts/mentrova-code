# app/services/llm.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Phi2LLM:
    def __init__(self, model_id="microsoft/phi-2"):
        print("Loading Phi-2 model...")
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map={"": self.device}
        )

    def generate(self, prompt: str, max_new_tokens: int = 150) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Global instance (loaded once at startup)
phi2 = Phi2LLM()

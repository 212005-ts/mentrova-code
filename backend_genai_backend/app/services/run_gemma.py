# app/services/run_gemma.py
import os
import threading
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch

# Load Hugging Face token from .env
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

class GemmaModel:
    def __init__(self):
        print("🔄 Loading Gemma model...")
        model_name = "google/gemma-2b"   # change to "google/gemma-7b" if needed

        # Auto-select device: MPS (Apple Silicon) → CUDA → CPU
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Use float16 on GPU/MPS for speed, float32 on CPU for stability
        dtype = torch.float16 if self.device != "cpu" else torch.float32

        # ✅ Pass Hugging Face token
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=HF_TOKEN,
            torch_dtype=dtype,
        ).to(self.device)

        print(f"✅ Gemma loaded successfully on {self.device.upper()}!")

    def generate(self, prompt: str, max_new_tokens=150):
        """Standard blocking generation (full output at once)."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # remove the prompt part (so response is clean)
        response = full_text[len(prompt):].strip()
        return response

    def stream(self, prompt: str, max_new_tokens=150):
        """Streaming generation: yields text chunks as they’re produced."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Hugging Face streaming utility
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        # Run model in background thread so we can yield tokens in real time
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            streamer=streamer
        )
        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # Yield each new text piece from the streamer
        for new_text in streamer:
            yield new_text

# Singleton instance
gemma = GemmaModel()

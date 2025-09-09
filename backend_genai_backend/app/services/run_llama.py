# app/services/run_llama.py
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
from threading import Thread

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

class Llama2Model:
    def __init__(self):
        print("🔄 Loading LLaMA2 model...")
        model_name = "meta-llama/Llama-2-7b-chat-hf"

        # Auto-select device: MPS (Apple Silicon) → CUDA → CPU
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Use float16 on GPU/MPS for speed, float32 on CPU
        dtype = torch.float16 if self.device != "cpu" else torch.float32

        # Load tokenizer + model with Hugging Face token
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=HF_TOKEN,
            torch_dtype=dtype,
        ).to(self.device)

        print(f"✅ LLaMA2 loaded successfully on {self.device.upper()}!")

    def generate(self, prompt: str, max_new_tokens=150):
        """Standard generation (returns full response at once)."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_text[len(prompt):].strip()
        return response

    def stream_generate(self, prompt: str, max_new_tokens=150):
        """Streaming generation (yields tokens progressively)."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            streamer=streamer
        )

        # Run generation in background thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # Yield tokens one by one
        for token in streamer:
            yield token

# Singleton (disabled in main.py for now)
llama2 = Llama2Model()

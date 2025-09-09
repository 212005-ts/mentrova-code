# app/services/run_qwen.py
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch, threading

class QwenModel:
    def __init__(self):
        print("🔄 Loading Qwen model...")
        model_name = "Qwen/Qwen2-1.5B-Instruct"   # you can swap for another Qwen2 variant

        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        dtype = torch.float16 if self.device != "cpu" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(self.device)

        print(f"✅ Qwen loaded successfully on {self.device.upper()}!")

    def generate(self, prompt: str, max_new_tokens=200):
        """Non-streaming generation (full text at once)."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text[len(prompt):].strip()

    def stream(self, prompt: str, max_new_tokens=200):
        """Streaming generation using TextIteratorStreamer."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        )

        # Run generation in a background thread so we can iterate streamer
        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for token in streamer:
            yield token

# singleton instance
qwen = QwenModel()

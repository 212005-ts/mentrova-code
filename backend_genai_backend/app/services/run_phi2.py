# app/services/run_phi2.py
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
import threading

class Phi3MiniModel:
    def __init__(self):
        print("🔄 Loading Phi-3 Mini (Instruct) model...")
        model_name = "microsoft/phi-3-mini-4k-instruct"

        # Auto-select device: MPS (Apple Silicon) → CUDA → CPU
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        dtype = torch.float16 if self.device in ["mps", "cuda"] else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
            ).to(self.device)
            print(f"✅ Phi-3 Mini (Instruct) loaded successfully on {self.device.upper()}!")
        except RuntimeError as e:
            if "MPS backend out of memory" in str(e):
                print("⚠️ MPS ran out of memory. Falling back to CPU...")
                self.device = "cpu"
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                ).to(self.device)
            else:
                raise

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _extract_assistant_reply(self, decoded_text: str) -> str:
        """
        Extract only the assistant's reply from the decoded chat sequence.
        """
        if "assistant" in decoded_text:
            # Keep everything after the last 'assistant'
            return decoded_text.split("assistant")[-1].strip(" :\n")
        return decoded_text.strip()

    def generate(self, prompt: str, max_new_tokens=150):
        """Standard blocking generation (full output at once)."""
        messages = [
            {"role": "system", "content": "You are Mentrova, an AI consultant."},
            {"role": "user", "content": prompt},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)

        try:
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        except RuntimeError:
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._extract_assistant_reply(decoded)

    def stream(self, prompt: str, max_new_tokens=150):
        """Streaming generation: yields text chunks as they’re produced."""
        messages = [
            {"role": "system", "content": "You are Mentrova, an AI consultant."},
            {"role": "user", "content": prompt},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,         # ✅ don’t include system/user text
            skip_special_tokens=True  # ✅ no <eos>/<pad>
        )

        generation_kwargs = dict(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            streamer=streamer,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        def run_generation():
            try:
                self.model.generate(**generation_kwargs)
            except RuntimeError:
                self.model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    streamer=streamer,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

        thread = threading.Thread(target=run_generation)
        thread.start()

        for new_text in streamer:
            yield new_text

# Singleton instance
phi2 = Phi3MiniModel()

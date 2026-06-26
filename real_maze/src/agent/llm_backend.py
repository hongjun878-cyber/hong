"""
LLM backend — thin wrapper around a local Qwen model.

Loads the model once (singleton) and exposes generate().
All agents share the same loaded weights; each call is independent.
"""

import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID   = "Qwen/Qwen3-8B"
_tokenizer = None
_model     = None

ACTION_WORDS = {"left", "up", "right", "down"}
WORD_TO_INT  = {"left": 0, "up": 1, "right": 2, "down": 3}


def load_model(model_id: str = MODEL_ID, device_map: str = "auto") -> None:
    global _tokenizer, _model
    if _model is not None:
        return
    print(f"[llm_backend] Loading {model_id} ...")
    _tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True,
    )
    _model.eval()
    print("[llm_backend] Model ready.")


def generate(
    prompt: str,
    system: str = "You are a maze navigation agent. Follow instructions exactly.",
    max_new_tokens: int = 32,
    temperature: float = 0.7,
    enable_thinking: bool = False,
) -> str:
    """
    Send a prompt to the model and return the raw text response.
    Qwen3 supports a /think toggle; we disable it for speed.
    """
    if _model is None:
        raise RuntimeError("Call load_model() before generate().")

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": prompt},
    ]
    text = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    inputs = _tokenizer([text], return_tensors="pt").to(_model.device)

    with torch.no_grad():
        out = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=_tokenizer.eos_token_id,
        )

    new_ids = out[0][inputs["input_ids"].shape[1]:]
    return _tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def parse_action(text: str, available: list) -> int:
    """
    Extract a valid action integer from free-form LLM output.
    Returns None if nothing valid can be parsed.
    """
    lower = text.lower()
    # exact word match first
    for word in ACTION_WORDS:
        if re.search(rf"\b{word}\b", lower):
            action = WORD_TO_INT[word]
            if action in available:
                return action
    return None

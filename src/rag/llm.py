from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rag import config

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using only the provided context. "
    "If the context does not contain the answer, say you don't know. "
    "Be concise and do not fabricate information."
)


@lru_cache(maxsize=1)
def _load_model():
    tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        config.LLM_MODEL,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    return tokenizer, model


def warm_up():
    """Load the model eagerly, e.g. at API startup, instead of on first request."""
    _load_model()


def generate(system_prompt: str, user_prompt: str, max_new_tokens: int = 400) -> str:
    tokenizer, model = _load_model()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = output[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def generate_answer(question: str, context: str, max_new_tokens: int = 400) -> str:
    return generate(
        SYSTEM_PROMPT,
        f"Context:\n{context}\n\nQuestion: {question}",
        max_new_tokens=max_new_tokens,
    )

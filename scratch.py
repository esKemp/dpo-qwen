import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", device)

tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).to(device)

msgs = [{"role": "user", "content": "In one sentence, what is direct preference optimization?"}]
inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(device)
out = model.generate(**inputs, max_new_tokens=60)
print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
from datasets import load_dataset

# A small, clean preference dataset: prompt / chosen / rejected
ds = load_dataset("trl-lib/ultrafeedback_binarized", split="train[:200]")

print(ds)
print("\n--- one example ---")
ex = ds[0]
for k in ex:
    print(f"\n[{k}]\n{ex[k]}")
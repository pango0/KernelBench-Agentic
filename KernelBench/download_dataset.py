from datasets import load_from_disk

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_from_disk("KernelBench")
print(ds)
import os
import pickle
import numpy as np

from datasets import load_dataset
import tiktoken

out_dir = os.path.dirname(__file__)
os.makedirs(out_dir, exist_ok=True)

# load wikitext-2 (raw)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

enc = tiktoken.get_encoding("gpt2")
eot = enc.eot_token  # end-of-text token id


def encode_split(split, out_name):
    print(f"Processing split: {split}")
    data = dataset[split]["text"]

    # drop empty lines and join
    texts = [t for t in data if t.strip()]
    text = "\n\n".join(texts)

    # GPT-2 BPE encode
    ids = enc.encode_ordinary(text)
    ids.append(eot)

    arr = np.array(ids, dtype=np.uint16)
    out_path = os.path.join(out_dir, out_name)
    arr.tofile(out_path)
    print(f"{split}: {len(ids)} tokens, {arr.nbytes/1e6:.2f} MB written to {out_path}")


# write train.bin and val.bin (NOTE: val, not validation)
encode_split("train", "train.bin")
encode_split("validation", "val.bin")

# minimal meta: just vocab_size
meta = {
    "vocab_size": enc.n_vocab,
}

with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

print("Done.")

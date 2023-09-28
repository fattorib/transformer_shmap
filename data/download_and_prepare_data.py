"""Downloads and prepares MiniPile dataset from 
`The MiniPile Challenge for Data-Efficient Language Models`
https://arxiv.org/abs/2304.08442

data preparation code borrowed from
https://github.com/karpathy/nanoGPT
"""

import os

import numpy as np
from tqdm import tqdm
from transformers import GPTNeoXTokenizerFast

from datasets import load_dataset

if __name__ == "__main__":
    num_proc = 6
    num_proc_load_dataset = num_proc

    dataset = load_dataset("JeanKaddour/minipile", num_proc=num_proc_load_dataset)

    tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")

    # this results in:
    # DatasetDict({
    #     train: Dataset({
    #         features: ['text'],
    #         num_rows: 1000000
    #     })
    #     validation: Dataset({
    #         features: ['text'],
    #         num_rows: 500
    #     })
    #     test: Dataset({
    #         features: ['text'],
    #         num_rows: 10000
    #     })
    # })

    def process(example):
        ids = tokenizer.encode(example["text"])
        ids.append(tokenizer.eos_token_id)
        out = {"ids": ids, "len": len(ids)}
        return out

    # tokenize the dataset
    tokenized = dataset.map(
        process,
        remove_columns=["text"],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        print(f"{split} - tokenized length: {arr_len}")
        filename = os.path.join(os.path.dirname(__file__), f"{split}.bin")
        dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
            # Batch together samples for faster write
            batch = dset.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            ).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    # train.bin is ~3GB, val.bin ~1.4MB
    # train has ~1.5B tokens (1,491,711,416)
    # val has ~0.6M tokens (693,668)

    # to read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')

import argparse
import json

import torch
import torch.nn.functional as F
import tqdm

from astrai.model import AutoModel
from astrai.tokenize import AutoTokenizer


def process_file(
    param_path: str, input_file: str, output_file: str, batch_size: int, text_key: str
):
    # Load model and tokenizer
    model = AutoModel.from_pretrained(param_path)
    tokenizer = AutoTokenizer.from_pretrained(param_path)
    model.to(device="cuda", dtype=torch.bfloat16)

    with open(input_file, "r", encoding="utf-8") as f:
        input_data = [json.loads(line) for line in f]

    texts = [item[text_key] for item in input_data]

    # Encode all texts
    print(f"Encoding {len(texts)} texts...")
    encoded_texts = [tokenizer.encode(text) for text in texts]

    output_data = []
    total_batches = (len(encoded_texts) + batch_size - 1) // batch_size

    for i in tqdm.tqdm(
        range(0, len(encoded_texts), batch_size),
        total=total_batches,
        desc="Computing perplexity",
    ):
        batch_encoded = encoded_texts[i : i + batch_size]
        batch_texts = texts[i : i + batch_size]

        # Find max length in batch and pad
        max_len = max(len(seq) for seq in batch_encoded)
        padded_ids = []
        masks = []

        for seq in batch_encoded:
            pad_len = max_len - len(seq)
            padded_seq = seq + [tokenizer.pad_id] * pad_len
            mask = [True] * len(seq) + [False] * pad_len
            padded_ids.append(padded_seq)
            masks.append(mask)

        # Convert to tensors
        input_ids = torch.tensor(padded_ids, device="cuda", dtype=torch.long)
        input_mask = torch.tensor(masks, device="cuda", dtype=torch.bool)

        # Compute perplexity
        output = model(input_ids, input_mask=input_mask)
        logits = output["logits"]

        # Shift for causal language modeling
        shifted_logits = logits[:, :-1, :]  # [batch_size, seq_len-1, vocab_size]
        shifted_input_ids = input_ids[:, 1:]  # [batch_size, seq_len-1]
        shifted_mask = input_mask[:, 1:]  # [batch_size, seq_len-1]

        # Compute cross entropy loss
        loss = F.cross_entropy(
            shifted_logits.flatten(0, 1),
            shifted_input_ids.flatten(0, 1),
            reduction="none",
        )

        loss = loss.view(shifted_input_ids.shape)  # [batch_size, seq_len-1]
        loss = loss * shifted_mask
        sentence_loss = loss.sum(dim=1) / shifted_mask.sum(dim=1).clamp(min=1)
        perplexity = torch.exp(sentence_loss)  # [batch_size]

        for text, ppl in zip(batch_texts, perplexity):
            output_data.append({text_key: text, "ppl": float(ppl.item())})

    # Write results
    with open(output_file, "w", encoding="utf-8") as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Perplexity computation complete. Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run perplexity with a Khaosz model.")
    parser.add_argument(
        "--param_path", type=str, required=True, help="Path to the model directory."
    )
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to the input file."
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to the output file."
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for evaluation."
    )
    parser.add_argument(
        "--text_key",
        type=str,
        default="text",
        help="Key for the text field in the input data.",
    )
    args = parser.parse_args()

    with torch.inference_mode():
        process_file(**vars(args))

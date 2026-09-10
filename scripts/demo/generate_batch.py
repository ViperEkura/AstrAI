from pathlib import Path

from astrai.inference import build_engine
from astrai.tokenize import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMETER_ROOT = Path(PROJECT_ROOT, "params")


def batch_generate():
    tokenizer = AutoTokenizer.from_pretrained(PARAMETER_ROOT)

    inputs = [
        "你好",
        "请问什么是人工智能",
        "今天天气如何",
        "我感到焦虑， 请问我应该怎么办",
        "请问什么是显卡",
    ]

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for q in inputs
    ]

    engine = build_engine(PARAMETER_ROOT)
    responses = engine.generate(
        prompt=prompts,
        stream=False,
        max_tokens=2048,
        temperature=0.8,
        top_p=0.95,
        top_k=50,
    )

    for q, r in zip(inputs, responses):
        print((q, r))


if __name__ == "__main__":
    batch_generate()

"""
Below codes are originally based on
https://github.com/zihangdai/xlnet/blob/master/prepro_utils.py
"""
import unicodedata


def preprocess_text(
        input_text: str,
        lower: bool=False,
        remove_space: bool=True,
        keep_accents: bool=False
) -> str:
    if remove_space:
        outputs = " ".join(input_text.strip().split())
    else:
        outputs = input_text
    outputs = outputs.replace("``", '"').replace("''", '"')

    if not keep_accents:
        outputs = unicodedata.normalize("NFKD", outputs)
        outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
    if lower:
        outputs = outputs.lower()

    return outputs

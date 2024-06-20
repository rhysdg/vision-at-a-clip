import gzip
import html
import os
from typing import Dict, List, Set, Union, Iterable

import ftfy
import numpy as np
import regex as re

from transformers import AutoProcessor, AutoModel

def default_bpe():
    """
    Taken from CLIP.
    https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py#L10
    BPE is the byte pair encoding that is used within the tokenizer. More can be read here:
    https://huggingface.co/course/chapter6/5?fw=pt
    Returns:
        the vocabulary byte-pair encodings

    """
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data/bpe_simple_vocab_16e6.txt.gz",
    )


def bytes_to_unicode() -> Dict[int, str]:
    """
    Taken from CLIP.
    https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py#L16
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word) -> Set[str]:
    """
    Taken from CLIP.
    https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py#L38
    Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text) -> str:
    """
    Taken from CLIP.
    https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py#L50
    """
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text) -> str:
    """
    Taken from CLIP.
    https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py#L56
    """
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


class Tokenizer(object):
    """
    Taken from CLIP.
    https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py#L62
    """

    def __init__(self, bpe_path: str = default_bpe(), device='cuda'):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")
        merges = merges[1 : 49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {
            "<|startoftext|>": "<|startoftext|>"device correction'
        
    
        self.siglip_processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-384", device=device)


    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text: str) -> List[int]:
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(
                self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" ")
            )
        return bpe_tokens

    def decode(self, tokens: List[int]) -> str:
        text = "".join([self.decoder[token] for token in tokens])
        text = (
            bytearray([self.byte_decoder[c] for c in text])
            .decode("utf-8", errors="replace")
            .replace("</w>", " ")
        )
        return text

    def encode_text(
        self,
        texts: Union[str, Iterable[str]],
        context_length: int = 77,
        truncate: bool = False,
        siglip: bool = False,
    ) -> np.array:
        """
        Taken from CLIP and reformatted to replace pytorch zeros with numpy zeros.
        Furthermore, this has been wrapped inside the Tokenizer class instead of being
        a separate function.
        https://github.com/openai/CLIP/blob/main/clip/clip.py#L197
        Returns the tokenized representation of given input string(s)
        Parameters
        ----------
        texts : Union[str, List[str]]
            An input string or a list of input strings to tokenize
        context_length : int
            The context length to use; all CLIP models use 77 as the context length
        truncate: bool
            Whether to truncate the text in case its encoding is longer than the context length
        siglip: bool
            Toggle silgip specific preprocessing
        Returns
        -------
        A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
        """

        if siglip:
            result = self.siglip_processor(text=texts, padding="max_length", return_tensors="np")['input_ids'].astype(np.int64)

        else:

            if isinstance(texts, str):
                texts = [texts]

            sot_token = self.encoder["<|startoftext|>"]
            eot_token = self.encoder["<|endoftext|>"]
            all_tokens = [[sot_token] + self.encode(text) + [eot_token] for text in texts]
            result = np.zeros((len(all_tokens), context_length), dtype=np.int32)

            for i, tokens in enumerate(all_tokens):
                if len(tokens) > context_length:
                    if truncate:
                        tokens = tokens[:context_length]
                        tokens[-1] = eot_token
                    else:
                        raise RuntimeError(
                            f"Input {texts[i]} is too long for context length {context_length}"
                        )
                result[i, : len(tokens)] = np.array(tokens)

        return result

"""Prepare the pinned TinyStories corpus and tokenizer."""
import hashlib
import json
from pathlib import Path

import numpy as np
import sentencepiece as spm

ROOT = Path(__file__).resolve().parent


def split_stories(text, fraction=.9):
    if not 0 < fraction < 1:
        raise ValueError('fraction must be between zero and one')
    stories = [s.strip() for s in text.split('<|endoftext|>') if s.strip()]
    split = int(fraction * len(stories))
    return stories[:split], stories[split:]


def join_tokens(stories, eos):
    joined = [eos]
    for story in stories:
        joined.extend(story)
        joined.append(eos)
    return np.asarray(joined, dtype=np.uint16)


def main():
    (ROOT/'artifacts').mkdir(parents=True,exist_ok=True)
    raw = ROOT/'data'/'TinyStoriesV2-GPT4-valid.txt'
    model = ROOT/'data'/'tok512.model'
    train, valid = split_stories(raw.read_text(encoding='utf-8'))
    if (len(train), len(valid)) != (24867, 2763):
        raise ValueError('story counts differ from the learning sheet')
    sp = spm.SentencePieceProcessor(model_file=str(model))
    if sp.vocab_size() != 512 or sp.eos_id() < 0:
        raise ValueError('tokenizer vocabulary or EOS does not match the learning sheet')
    encoded_train = sp.encode(train, out_type=int, num_threads=8)
    encoded_valid = sp.encode(valid, out_type=int, num_threads=8)
    decoded = sp.decode(encoded_valid, num_threads=8)
    train_tokens = join_tokens(encoded_train, sp.eos_id())
    valid_tokens = join_tokens(encoded_valid, sp.eos_id())
    for name, tokens in [('train', train_tokens), ('valid', valid_tokens)]:
        with (ROOT/'artifacts'/f'{name}-tokens.npy').open('xb') as f:
            np.save(f, tokens)
    def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
    sources = sorted((ROOT/'data').iterdir())
    meta = dict(corpus_revision='f54c09fd23315a6f9c86f9dc80f725de7d8f9c64',
                tokenizer_revision='0bd21da', source_sha256={p.name:digest(p) for p in sources if p.is_file()},
                train_stories=len(train), valid_stories=len(valid), eos=sp.eos_id(), vocab=sp.vocab_size(),
                train_tokens=len(train_tokens), valid_tokens=len(valid_tokens),
                train_characters=sum(map(len, train)), valid_characters=sum(map(len, valid)),
                valid_decode_mismatches=[i for i,(a,b) in enumerate(zip(valid, decoded)) if a != b],
                convention='leading EOS and one EOS after every story; character counts exclude delimiters')
    with (ROOT/'artifacts'/'data-manifest.json').open('x') as f:
        json.dump(meta, f, indent=2); f.write('\n')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()

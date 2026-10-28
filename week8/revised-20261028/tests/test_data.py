import numpy as np
from pipeline import split_stories, join_tokens


def test_split_strips_and_drops_empty_stories_before_splitting():
    train, valid = split_stories(' a <|endoftext|>  <|endoftext|> b <|endoftext|> c ', fraction=2/3)
    assert train == ['a', 'b']
    assert valid == ['c']


def test_join_tokens_uses_eos_as_each_story_boundary():
    result = join_tokens([[7, 8], [9]], eos=2)
    np.testing.assert_array_equal(result, [2, 7, 8, 2, 9, 2])

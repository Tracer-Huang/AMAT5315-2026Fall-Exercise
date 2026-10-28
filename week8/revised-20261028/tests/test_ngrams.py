import numpy as np
from ngrams import fit_counts, score_tokens, next_distribution, next_with_short_context


def test_bigram_counts_and_add_k_probabilities():
    train = np.array([2, 1, 1, 2, 1], dtype=np.uint16)
    model = fit_counts(train, n=2, vocab=4, k=.1, eos=2)
    probs, seen = score_tokens(model, np.array([2, 1, 2], dtype=np.uint16))
    np.testing.assert_allclose(probs, [2.1/2.4, 1.1/2.4])
    assert seen.tolist() == [True, True]


def test_unseen_context_is_uniform_and_counted_as_unseen():
    model = fit_counts(np.array([2, 1, 2], dtype=np.uint16), n=3, vocab=4, k=.01, eos=2)
    probs, seen = score_tokens(model, np.array([2, 3, 3, 1], dtype=np.uint16))
    assert np.all(probs[1:] == .25)
    assert not seen[1:].any()


def test_next_token_distribution_is_normalized_and_uses_last_context():
    model = fit_counts(np.array([2, 1, 1, 2, 1], dtype=np.uint16), n=2, vocab=4, k=.1, eos=2)
    p = next_distribution(model, [2, 1])
    np.testing.assert_allclose(p, [.1/2.4, 1.1/2.4, 1.1/2.4, .1/2.4])
    assert np.isclose(p.sum(), 1.)


def test_trigram_generation_starts_with_available_bigram_context():
    seq = np.array([2, 1, 1, 2, 1], dtype=np.uint16)
    models = {n: fit_counts(seq, n=n, vocab=4, k=.1, eos=2) for n in (1, 2, 3)}
    np.testing.assert_allclose(next_with_short_context(models, 3, [2]), next_distribution(models[2], [2]))
    np.testing.assert_allclose(next_with_short_context(models, 3, [2, 1]), next_distribution(models[3], [2, 1]))

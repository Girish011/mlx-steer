from mlx_steer.monitor.repetition import compute_repetition_score


def test_repetition_empty_is_zero():
    assert compute_repetition_score("") == 0.0


def test_repetition_no_repeat_is_low():
    text = "one two three four five six seven"
    assert compute_repetition_score(text, ngram=2) == 1.0 / 6.0


def test_repetition_loop_is_high():
    text = "a b a b a b a b a b a b"
    assert compute_repetition_score(text, ngram=2) > 0.5


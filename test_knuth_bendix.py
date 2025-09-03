from knuth_bendix import knuth_bendix_completion, run_rules


def test_overlap_completion():
    initial = {("ab", "ba"), ("bb", "b")}
    rules = knuth_bendix_completion(initial)
    assert run_rules("bba", rules) == run_rules("bab", rules) == "ab"

from knuth_bendix import knuth_bendix_completion, run_rules


def test_overlap_completion(initial):
    rules = knuth_bendix_completion(initial)
    return rules

import random

alphabet_num = 25

alphabet = tuple(chr(i + 65) for i in range(alphabet_num))

# ルールは幾何分布を使ってランダムに生成
rule_num = 10
rule_length_expect_inv = 0.2 #幾何分布のパラメータpにあたる

rule = []
while len(rule) < rule_num:
    i = 0
    while random.random() > rule_length_expect_inv:
        i += 1
    j = 0
    while random.random() > rule_length_expect_inv:
        j += 1
    before = ''.join(random.choices(alphabet, k=i))
    after = ''.join(random.choices(alphabet, k=j))
    strings = [before, after]
    strings.sort(key=lambda s: (len(s), s), reverse=True)
    rule.append(tuple(strings))

print(rule)
initial_rule = set(sorted(rule, key=lambda k: len(k[0]), reverse=True))

rules = knuth_bendix_completion(initial_rule)
print(rules)
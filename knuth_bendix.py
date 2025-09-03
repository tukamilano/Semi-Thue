from typing import Iterable, List, Set, Tuple

Rule = Tuple[str, str]


def run_rules(s: str, rules: Iterable[Rule]) -> str:
    """Reduce string ``s`` by applying ``rules`` until no change."""
    old_s = None
    while s != old_s:
        old_s = s
        for lhs, rhs in rules:
            s = s.replace(lhs, rhs)
    return s


def reduce_rules(equations: Iterable[Rule]) -> Set[Rule]:
    """Orient and simplify a set of equations into rewrite rules."""
    R: Set[Rule] = set()
    E = sorted(equations, key=lambda k: len(k[0]), reverse=True)
    while E:
        a, b = E.pop()
        a = run_rules(a, R)
        b = run_rules(b, R)
        if a == b:
            continue
        if (len(b), b) > (len(a), a):
            R.add((b, a))
        if (len(a), a) > (len(b), b):
            R.add((a, b))
    return R


def critical_pairs(r1: Rule, r2: Rule) -> List[Rule]:
    """Compute critical pairs between two rules considering overlaps."""
    l1, r1_rhs = r1
    l2, r2_rhs = r2
    pairs: List[Rule] = []

    max_overlap = min(len(l1), len(l2))
    # l1 suffix overlaps l2 prefix
    for i in range(1, max_overlap + 1):
        if l1[-i:] == l2[:i]:
            s = r1_rhs + l2[i:]
            t = l1[:-i] + r2_rhs
            pairs.append((s, t))
    # l2 suffix overlaps l1 prefix
    for i in range(1, max_overlap + 1):
        if l2[-i:] == l1[:i]:
            s = r2_rhs + l1[i:]
            t = l2[:-i] + r1_rhs
            pairs.append((s, t))
    # inclusion: l1 contains l2
    if l2 in l1:
        idxs = [i for i in range(len(l1) - len(l2) + 1) if l1[i:i+len(l2)] == l2]
        for idx in idxs:
            s = l1[:idx] + r2_rhs + l1[idx + len(l2):]
            t = r1_rhs
            pairs.append((s, t))
    # inclusion: l2 contains l1
    if l1 in l2:
        idxs = [i for i in range(len(l2) - len(l1) + 1) if l2[i:i+len(l1)] == l1]
        for idx in idxs:
            s = l2[:idx] + r1_rhs + l2[idx + len(l1):]
            t = r2_rhs
            pairs.append((s, t))

    return pairs


def knuth_bendix_completion(equations: Iterable[Rule]) -> Set[Rule]:
    """Complete a semi-Thue system using the Knuth-Bendix algorithm."""
    E: Set[Rule] = set(equations)
    R = reduce_rules(E)
    changed = True
    while changed:
        changed = False
        new_eqs: Set[Rule] = set()
        rules_list = list(R)
        for i in range(len(rules_list)):
            for j in range(i, len(rules_list)):
                for s, t in critical_pairs(rules_list[i], rules_list[j]):
                    s_norm = run_rules(s, R)
                    t_norm = run_rules(t, R)
                    if s_norm != t_norm:
                        new_eqs.add((s_norm, t_norm))
        if new_eqs:
            R |= reduce_rules(new_eqs)
            changed = True
    return R

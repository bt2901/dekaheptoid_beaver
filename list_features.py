from .list_representation import tape2list
from .common import cnt2gray, rightmost_index_where, leftmost_index_where
from itertools import groupby

def is_power_of_two(x):
    return (x & (x - 1)) == 0

def closest_power_of_two(x):
    possible_results = floor(log(x, 2)), ceil(log(x, 2))
    for candidate in possible_results :
        if abs(2 ** candidate - x) <= 1:
            return candidate
    return None


def describe_list(thelist):
    gray = cnt2gray(thelist)
    last = thelist[-1]
    yield f"last %2 == {last%2}"
    if last <= 1:
        yield f"last == {last}"
    else:
        yield f"last == 2+"
    yield f"L == {len(thelist)}"
    p = len(thelist) % 2
    yield f"L %2 == {p}"

    p = gray % 2
    yield f"gray %2 == {p}"
    if gray <= 1:
        yield f"gray == {gray}"
    if 2**(len(thelist)-1) -1 == gray:
        yield f"gray-adic == 2^L - 1"
    else:        
        for cand, text in [(gray-1, " + 1"), (gray, ""), (gray+1, " - 1")]:
            if is_power_of_two(cand):
                yield f"gray-adic == 2^k{text}"
    if all(x % 2 == 0 for x in thelist):
        yield "all even"
    try:
        rightmost_odd_idx = rightmost_index_where(lambda i: thelist[i] % 2, thelist)
        yield f"rightmost_odd_idx == {rightmost_odd_idx}"
    except StopIteration:
       yield "rightmost_odd_idx == -1"
    try:
        rightmost_odd_idx_alt = rightmost_index_where(lambda i: thelist[i] % 2, thelist[:-1])
        yield f"rightmost_odd_idx_alt == {rightmost_odd_idx_alt}"
    except StopIteration:
       yield "rightmost_odd_idx_alt == -1"

    try:
        num_leading_zeros = leftmost_index_where(lambda i: thelist[i] != 0, thelist)
        yield f"num_leading_zeros == {num_leading_zeros}"
    except StopIteration:
        yield "num_leading_zeros == -1"
    if 0 in thelist[1:-1]:
        internal_zeros_len = max(len(list(g)) for k, g in groupby(thelist[1:-1], lambda x: x == 0) if k)
    else:
        internal_zeros_len = 0
    yield f"internal_zeros_len == {internal_zeros_len}"
    yield f"total_sum == {sum(thelist)}"
    yield f"num_nans == {sum(1 for x in thelist if str(x) == 'nan')}"



def test_gray_numbers(gray_numbers, sign):
    seqs = []
    beg, end = None, None
    for prev, cur in zip(gray_numbers[:-1], gray_numbers[1:]):
        if cur - prev == sign:
            if beg is None:
                beg = prev
            end = cur
        else:
            if beg is not None and abs(end-beg) > 1:
                seqs.append((beg, end))
            beg, end = None, None
    stats = [abs(end-beg) for beg, end in seqs]
    return (len(stats), sum(stats), max(stats or [float("-inf")]))


def measure_prop_of_seqs(tapes_history, allowed_states, is_mirrored=False, verbose=False):

    lists = [
        tape2list("".join(raw_tape[::-1]) if is_mirrored else raw_tape) 
        for i, raw_tape, state in tapes_history 
        if state in allowed_states and raw_tape
    ]
    gray_numbers = [cnt2gray(thelist) for thelist in lists]
    if not len(gray_numbers):
        return 0.0
    pos_stats = test_gray_numbers(gray_numbers, 1)
    neg_stats = test_gray_numbers(gray_numbers, -1)

    perc = (pos_stats[1] + neg_stats[1]) / len(gray_numbers)
    if verbose:
        print(f"""\
            Out of {len(gray_numbers)} configs: {pos_stats[0] + neg_stats[0]} monotonic sequences found. 
            Total length is {pos_stats[1] + neg_stats[1]} ({100*perc:.2f}%)
            (max len of increasing seq is {pos_stats[2]}, of decreasing seq is {neg_stats[2]})
        """)

    return perc

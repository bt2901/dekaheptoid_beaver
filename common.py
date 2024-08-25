from itertools import groupby
import string

def cnt2gray(thelist, is_mirrored=False):
    if is_mirrored:
        thelist = thelist[::-1]
    thelist = thelist[:-1]
    if any(isinstance(e, float) for e in thelist):
        return float("nan")
    num = sum([(x % 2) * 2 ** i for i, x in enumerate(thelist[::-1])])
    mask = num
    while mask:
        mask >>= 1
        num ^= mask
    return num

def leftmost_index_where(predicate, thelist):
    list_indices = range(len(thelist))
    return next(filter(
        predicate,
        list_indices
    ))

def rightmost_index_where(predicate, thelist):
    list_indices = range(len(thelist))
    return next(filter(
        predicate,
        reversed(list_indices)
    ))

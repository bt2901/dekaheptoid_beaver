from .common import cnt2gray, rightmost_index_where, leftmost_index_where


def is_increment_safe(thelist, is_mirrored=False):
    if is_mirrored:
        thelist = thelist[::-1]
    gray = cnt2gray(thelist)
    last = thelist[-1]
    if len(thelist) <= 1 or last < 1:
        return False
    if all(x % 2 == 0 for x in thelist):
        return False
    rightmost_odd_idx = rightmost_index_where(lambda i: thelist[i] % 2, thelist)
    if rightmost_odd_idx == 0:
        # overflow
        return False
    return True

def apply_increment(thelist, is_mirrored=False):
    rightmost_odd_idx = rightmost_index_where(lambda i: thelist[i] % 2, thelist)

    newlist = list(thelist)
    newlist[rightmost_odd_idx - 1] += 1
    newlist[-1] -= 1
    if is_mirrored:
        newlist = newlist[::-1]
    return newlist


#            if len(thelist) % 2 == 0:
#                newlist = [1] + thelist
#            else:
#                newlist = [1] + thelist
#                #if thelist[0] % 2 == 0:
#                #    newlist = [1] + thelist
#                #else:
#                #    newlist[0] += 2

def apply_unsafe_increment_1N(thelist, is_mirrored=False):
    newlist = list(thelist)

    newlist[0] += 1
    newlist[-1] -= 1
    if is_mirrored:
        newlist = newlist[::-1]

    return newlist

def apply_shallow_increment(thelist):
    newlist = list(thelist)
    newlist[-2] += 1
    newlist[-1] -= 1
    return newlist

def apply_excluding_increment(thelist):
    last = thelist[-1]
    newlist = list(thelist)
    rightmost_odd_idx = rightmost_index_where(lambda i: thelist[i] % 2, thelist[:-1])
    if rightmost_odd_idx == 0:
         # weird overflow
         return []
    newlist[rightmost_odd_idx - 1] += 1
    newlist[-1] -= 1
    return newlist

def apply_increment_D_even(thelist):
    last = thelist[-1]
    newlist = list(thelist)
    rightmost_odd_idx = rightmost_index_where(lambda i: thelist[i] % 2, thelist[:-1])
    if rightmost_odd_idx == 0:
         # weird overflow
         return []
    newlist[rightmost_odd_idx] += 1
    newlist[-1] -= 1
    return newlist

def apply_halt(thelist):
    return "HALT"

def apply_overflow(thelist):
    newlist = list(thelist)
    newlist[0] += 1
    return [0] + newlist

def apply_weird_overflow(thelist):
    newlist = list(thelist)
    newlist[1] += 1
    newlist[2] += 1

    return newlist

def apply_weird_double_zero(thelist):
    return apply_weird_zero_n(thelist, 2)

def apply_weird_one_zero(thelist):
    return apply_weird_zero_n(thelist, 1)

def apply_weird_zero_n(thelist, num_add_zeros):
    newlist = [0] * num_add_zeros + list(thelist)
    newlist = apply_leading_zero_increment(newlist)

    return newlist

def apply_leading_zero_increment(thelist):
    newlist = list(thelist)
    num_leading_zeros = leftmost_index_where(lambda i: newlist[i] != 0, newlist)
    newlist[num_leading_zeros - 1] += 1
    newlist[-1] -= 1
    return newlist


def apply_nan_safe_increment(thelist):
    if any(str(x) == '-inf' for x in thelist):
        div_index = rightmost_index_where(lambda i: thelist[i] == float("-inf"), thelist)
        newlist = thelist[:div_index] + apply_increment(thelist[div_index:])
        return newlist
    return []

def apply_zero(thelist):
    first, last = thelist[0], thelist[-1]
    return [0, 0] + [first + 1] + thelist[1:-1] + [last - 1]

def apply_zero_D(thelist):
    return(apply_unsafe_increment_1N(apply_overflow(thelist)))
    # first, last = thelist[0], thelist[-1]
    # return [1] + [first + 1] + thelist[1:-1] + [last - 1]

#def apply_overflow_variant(thelist):
#    first, last = thelist[0], thelist[-1]
#    return [0, 0, 1] + [first + 1] + thelist[1:-1] + [last - 1]

def apply_zero_variant(thelist):
    return [0, 0] + thelist

def apply_zero_increment(thelist, N=1):
        first, last = thelist[0], thelist[-1]
        return [0] + [first + 1 + N] + thelist[1:-1] + [last - 1]

def apply_init(thelist):
        # equivalent to "apply_halve o apply_zero_variant" lol
        if len(thelist) == 0: return [0]
        first, last = thelist[0], thelist[-1]
        return [0] + [first + 1]

def apply_halve(thelist):
    newlist = list(thelist)
    newlist = newlist[:-1]
    return newlist

def apply_halve_and_increment(thelist):
    newlist = list(thelist)
    newlist = newlist[:-1]
    newlist[-1] += 1
    return newlist

def apply_weird_incrementing_halve(thelist):
    newlist = list(thelist)
    if len(thelist) <= 4:
        newlist[1] += 2
    else:
        newlist[1] += 3
    return newlist

def apply_weird_halve_zero(thelist):
    newlist = thelist[:-2]
    newlist[0] += 1
    newlist[-2] += 1
    return [0, 0] + newlist

def apply_weird_halve_zero_variant(thelist):
    newlist = thelist[:-2]
    newlist[0] += 1
    newlist[-1] += 1
    return [0, 0] + newlist

def apply_weird_recursive_increment(thelist):
    if thelist[-1] == 0:
        newlist = [0] + thelist[:-1]
        newlist[-1] += 1
    else:
        newlist = [0] + thelist
    while newlist[-1] > 1:
        newlist = apply_unsafe_increment_1N(newlist)
    newlist = [0] + newlist
    newlist = apply_halve(newlist)
    newlist[-2] += 1
    newlist[2] -= 3 + (thelist[-1] == 0)
    return newlist


def apply_smudge_internal_zeros(thelist):
    newlist = (apply_unsafe_increment_1N(thelist))
    just_smudged = False
    for i in range(len(newlist) - 2):
        if just_smudged:
            just_smudged = False
            continue
        if newlist[i] > 0 and newlist[i+1] == 0:
            newlist[i] -= 1 
            newlist[i+1] += 1
            just_smudged = True
    if newlist[-1] < 0:
        newlist = newlist[:-1]
    return newlist

def apply_reverse_smudge_internal_zeros(thelist):
    newlist = (apply_unsafe_increment_1N(thelist))
    # if newlist[-1] <= 0:
    #    num_ending_zeros = rightmost_index_where(lambda i: newlist[i] > 0, newlist)
    # else:
    #    num_ending_zeros = len(newlist)

    newlist2 = []
    just_smudged = False
    for i in reversed(range(0, len(newlist))):
        if just_smudged:
            just_smudged = False
            continue
        if newlist[i] > 0 and newlist[i-1] == 0:
            newlist2 += [newlist[i] - 1] + [float("-inf")]*3
            just_smudged = True
        else:
            newlist2 += [newlist[i]]
    newlist2 = list(reversed(newlist2))
    #nan_split_idx = rightmost_index_where(lambda i: newlist[i] == 0 and i < num_ending_zeros, newlist)
    #print(nan_split_idx)
    #newlist[nan_split_idx+1] -= 1
    #return newlist[:nan_split_idx] + [float("-inf")]*3 + newlist[nan_split_idx+1:]
    if newlist2[-1] < 0:
        newlist2 = newlist2[:-1]
        if newlist2[-2:] == [0, 0]:
            newlist2 = newlist2[:-2] + [float("-inf")]*3
    return newlist2


def try_increment(thelist, is_mirrored=False):
    if not is_increment_safe(thelist):
        return None
    return apply_increment(thelist, is_mirrored)

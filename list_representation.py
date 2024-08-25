from itertools import groupby
import string

def tape2list_new(raw_tape):
    raw_tape = raw_tape.replace("10", "A")
    thelist = []
    for g in raw_tape.split("1"):
        cur_len = len(g)
        if "0" in g:
            # should not happen if the tape in legal state
            for sub_g in g.split("0"):
                thelist.append(len(sub_g))
                thelist.append(float("nan"))
        else:
            thelist.append(cur_len)
    return thelist

def tape2list(raw_tape):
    raw_tape = raw_tape.replace("10", "A")
    thelist = []
    if raw_tape[0] == '1':
        thelist.append(0)
    for k, g in groupby(raw_tape):
        cur_len = len(list(g))
        # assert k in ['A', '1']
        if k == '1':
            for i in range(cur_len - 1):
                thelist.append(0)
        elif k == 'A':
            thelist.append(cur_len)
        else:
            # should not happen if the tape in legal state
            # print(k)
            thelist += [float("-inf")] * cur_len
    if raw_tape[-1] == '1' and len(raw_tape) > 1:
        thelist.append(0)
    return thelist

def get_raw_tape_from_chained_tape(line, block_size=4, unit="10"):
    inner_data = line.split("0" * block_size + "^inf")[1].strip().strip("0")
    divided = inner_data.split()

    divided = [d.strip("()") for d in divided if not (">" in d or "<" in d)]
    exploded = [
        one_block for p in divided
        for one_block, _, exponent in [p.partition("^")]
        for _ in range(int(exponent or 1))
    ]
    return "".join(exploded).replace(unit, "A")

def get_state_from_chained_tape(line, block_size):
    inner_data = line.split("0" * block_size + "^inf")[1].strip().strip("0")
    divided = inner_data.split()
    return [d for d in divided if (">" in d or "<" in d)][0]

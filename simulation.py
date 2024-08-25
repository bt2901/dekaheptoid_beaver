from Visual_Simulator import IO, DirectSimulator, parse_config, should_print
import string

def init_sim(tm, start_config):
    tm = IO.get_tm(tm)
    sim = DirectSimulator(tm)

    if start_config:
        state, left, right = parse_config(start_config)
    else:
        state = 0
        left = []
        right = []

    sim.state = state
    for i, symb in enumerate(left):
        # Write left half of tape so that the rightmost symbol is at location -1.
        sim.tape.write_pos(symb, i-len(left))
    for i, symb in enumerate(right):
        # Write right half of tape so that the leftmost symbol is at location 0.
        sim.tape.write_pos(symb, i)
    return sim

def print_raw_tape(sim, should_strip=True):
    raw_tape = []
    print_range = range(sim.tape.pos_leftmost(), sim.tape.pos_rightmost()+1)

    for pos in print_range:
        value = sim.tape.read_pos(pos)
        raw_tape.append(value)
    raw_tape = "".join(str(x) for x in raw_tape)
    if should_strip:
        if sim.tape.pos_leftmost() == sim.tape.position:
            raw_tape = raw_tape.rstrip("0")[1:]
        if sim.tape.pos_rightmost() == sim.tape.position:
            raw_tape = raw_tape.lstrip("0")[:-1]
    return raw_tape

def run_direct_sim(tm, n_steps=1024, start_config=None, give_up_limit=1024, print_opts="r", selected_state=None):
    sim = init_sim(tm, start_config)

    tapes_history = []
    cur_step = 0
    while not sim.halted:
        if should_print(sim, print_opts):
            if selected_state is None or string.ascii_uppercase[sim.state] == selected_state:
                raw_tape = print_raw_tape(sim)
                tapes_history.append((cur_step, raw_tape, string.ascii_uppercase[sim.state]))
            i = 0
        if len(tapes_history) >= n_steps:
            break
        sim.step()
        cur_step += 1
        i += 1
        if i > give_up_limit:
            tapes_history.append((cur_step, "TimeLimit Exceeded!", "TLE"))
            break

    return tapes_history



def skip_n_increments(thelist, k, reverse_count=False):
    newlist = list(thelist)
    old_n = cnt2gray(newlist)
    n = old_n + k
    if reverse_count:
        n = old_n - k
    denom = 2**(len(thelist) - 1)
    for i, oldval in enumerate(thelist):
        if i == len(thelist) - 1:
            newlist[i] -= k
        else:
            newlist[i] += abs(int(0.5 + old_n/denom) - int(0.5 + n/denom))
        denom //= 2
        
    return newlist

if __name__ == "__main__":

    with open("uniq_tms.txt", "r") as f:
        UNIQ_TMS = f.read().split("\n")

    for one_tm_string in UNIQ_TMS:
        if one_tm_string not in simulator_params:
            continue
        print(one_tm_string)
        print_opts, the_state = simulator_params[one_tm_string]
        tapes_history = run_direct_sim(one_tm_string, 4096 * 4 * 12, give_up_limit=10**6, print_opts=print_opts)
        print(len(tapes_history))
        allowed_states = [the_state]
        tape_evolution[one_tm_string] = []
        for i, raw_tape, state in tapes_history:
            if state in allowed_states and raw_tape:

                if print_opts == "l":
                    thelist = tape2list("".join(raw_tape[::-1]))
                else:
                    thelist = tape2list(raw_tape)
                tape_evolution[one_tm_string].append(thelist)
        with open(one_tm_string + ".pkl", "wb") as f:
            pickle.dump(tape_evolution[one_tm_string], f)
        with open(one_tm_string + ".tape.pkl", "wb") as f:
            pickle.dump(tapes_history, f)

    success_count = Counter()

    for one_tm_string in tape_evolution.keys():
        if one_tm_string == '1RB1LC_1RC0RF_0LD1RE_0LA1LD_0RC0RB_0RE---':
            continue
        cnt = Counter()
        predicted = []
        countdown = 0
        k = 0
        pred_step = 0
        old_config = None
        for step, config in tape_evolution[one_tm_string]:
            countdown -= 1
            k += 1
            if not predicted: 
                if config[-1] > 1:
                    gray_val = cnt2gray(config)
                    reverse_flag = True if one_tm_string in [
                        "1RB1LC_1RC---_0LD1RE_0LA1LD_0RF0RB_0LC1RE", 
                        "1RB1LC_1RC---_0LD1RF_0LE1LE_0LA1LE_0RC0RB",
                    ] else False
                    sigma = sum(config) % 2
                    if sigma == 0:
                        reverse_flag = not reverse_flag

                    # TODO: might be more fine-grained, depending on reverse_flag
                    safe_amount_of_steps = min(
                        config[-1] - 1,
                        2**(len(config) - 1) - 1 - gray_val,
                        2**(len(config) - 1) - 1 - config[-1],
                        gray_val
                    )
                    predicted = [skip_n_increments(config, k, reverse_flag) for k in range(safe_amount_of_steps)]
                    if len(predicted):
                        countdown = safe_amount_of_steps - 1
                        old_config = config
                        k = 0
                        pred_step = step
            elif k == 1:
                # did we manage to at least predict one increment correctly?
                result_weak = (predicted[1] == config)

            if countdown == 0:
                result = (predicted[-1] == config)
                if not result and result_weak:
                    print(one_tm_string)
                    print([(p, cnt2gray(p)) for p in predicted])
                    print(config, cnt2gray(config))
                success_count[(one_tm_string, result)] += 1
                predicted = []

    print(success_count)

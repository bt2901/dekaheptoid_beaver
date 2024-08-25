from Visual_Simulator import IO, DirectSimulator, parse_config, should_print
import string

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
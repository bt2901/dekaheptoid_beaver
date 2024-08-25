from dekaheptoid_beaver.common import cnt2gray, rightmost_index_where
from dekaheptoid_beaver.list_representation import tape2list
from dekaheptoid_beaver.simulation import print_raw_tape, run_direct_sim, init_sim

from Visual_Simulator import IO, DirectSimulator, parse_config, should_print

import json


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


simulator_params = {}

with open("all_tms.txt", "r") as f:
    ALL_TMS = f.read().split("\n")

for one_tm_string in ALL_TMS:
    print()
    print(one_tm_string)
    states = IO.get_tm(one_tm_string).list_base_states()
    for print_opts in ['l', 'r']:
        tapes_history = run_direct_sim(one_tm_string, 256, give_up_limit=10**3, print_opts=print_opts)
        # print(print_opts, len(tapes_history), tapes_history[-1][0], tapes_history[-1][-1])
        if tapes_history[-1][2] == "TLE":
            print("TLE between", tapes_history[-2][0], "and", tapes_history[-1][0])
        max_perc = 0
        best_comb = []
        
        for comb in powerset([string.ascii_uppercase[s] for s in states]):
            if len(comb) > 1:
                continue
            perc = measure_prop_of_seqs(tapes_history, comb, is_mirrored=(print_opts=="l"))
            if perc > max_perc:
                max_perc = perc
                best_comb = comb
        # in theory, the TM could have more than one end-going state; in practice, it's always a single state
        if len(best_comb) == 1:
            best_comb = best_comb[0]
        if max_perc > 0.4:
            print(max_perc)
            if print_opts == "l":
                print(f"Assuming form: <{best_comb} (10)* (1 (10)*)* ")
            else:
                print(f"Assuming form: (10)* (1 (10)*)* {best_comb}>" )
            simulator_params[one_tm_string] = (print_opts, best_comb)

with open("simulator_params.json", "w", encoding="utf8") as f:
    json.dump(simulator_params, f)


initial_config = "0"
for one_tm_string in ALL_TMS:
        if one_tm_string not in simulator_params:
            continue
    
        orient, start_state = simulator_params[one_tm_string]
        if orient == "l":
            start_config = initial_config[::-1] + " <" + start_state
        else:
            start_config = start_state + "> " + initial_config

        sim = init_sim(tm, start_config)

        sim.step()
        back_states[one_tm_string] = string.ascii_uppercase[sim.state]


for one_tm_string in ALL_TMS:
    if one_tm_string not in simulator_params:
        continue
    
    orient, start_state = simulator_params[one_tm_string]
    back_state = back_states[one_tm_string]
    print()
    print("==", one_tm_string, "==")
    print()
    for initial_config, initial_state, initial_dir in [("1010", start_state, ">"), ("1010", back_state, "<")]:
        if initial_dir == ">":
            if orient == "l":
                start_config = initial_config[::-1] + " <" + initial_state
            else:
                start_config = initial_state + "> " + initial_config
        else:
            if orient == "l":
                start_config = initial_state[::-1] + "> " + initial_config
            else:
                start_config = initial_config + " <" + initial_state

        sim = init_sim(tm, start_config)

        for i in range(2 * len(start_config)):
            raw_tape = print_raw_tape(sim, should_strip=False)
            # print(raw_tape, sim.tape.position, string.ascii_uppercase[sim.state])
            sim.step()
            if (sim.tape.position in (sim.tape.pos_leftmost(), sim.tape.pos_rightmost())):
                raw_tape = print_raw_tape(sim, should_strip=False)
                # print(raw_tape, sim.tape.position, string.ascii_uppercase[sim.state])
                sim.step()
                if initial_state == string.ascii_uppercase[sim.state]:
                    break

        raw_tape = print_raw_tape(sim, should_strip=False)
        if (initial_config in raw_tape):
            print(start_config, "==>", " ".join(start_config.split()[::-1]))
            print("in", sim.step_num, "steps")
        # print(sim.tape.position, string.ascii_uppercase[sim.state])

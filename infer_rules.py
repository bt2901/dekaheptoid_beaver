from dekaheptoid_beaver.common import cnt2gray, rightmost_index_where
from dekaheptoid_beaver.list_representation import tape2list
from dekaheptoid_beaver.simulation import print_raw_tape, run_direct_sim, init_sim

from Visual_Simulator import IO, DirectSimulator, parse_config, should_print

import json

ALL_CONDS = [
        # skelet 17 inspired
        'last == "0" and rightmost_odd_idx_alt != -1 and rightmost_odd_idx != 0',
        # 'last == "0" and rightmost_odd_idx != 0 and rightmost_odd_idx_alt != -1 and `gray-adic` != "2^L - 1"', 
            'last == "0" and (rightmost_odd_idx <= 0) and `gray-adic` != "2^L - 1" and num_leading_zeros != 2',
        #'`gray-adic` == "2^L - 1" and (rightmost_odd_idx == 0 or all_zeros == 1) and last == "0"', 
        #'`gray-adic` == "2^L - 1" and (rightmost_odd_idx == 0 or all_zeros == 1) and last == "2+"', 
        '`gray-adic` == "2^L - 1" and rightmost_odd_idx == 0 and last == "0"', 
        '`gray-adic` == "2^L - 1" and rightmost_odd_idx == 0 and last == "2+"', 
        #'last != "0" and all_even == 1 and `gray-adic` != "2^L - 1"',
        'last != "0" and all_even == 1 and rightmost_odd_idx_alt == -1',
        # 'last != "0" and all_even == 0 and rightmost_odd_idx != 0',
        """last != "0" and rightmost_odd_idx > 0""",
    
        # _0LC1RF_0LD1LC_1RE1LB_0RE1LA_0RB0RA
        'all_even == 1',
        # 1RB1LC_1RC---_0LD1RE_0LA1LD_0RC1RF_0LF0RB
        'num_leading_zeros == 2 and last == "0" and all_even == 1',
        # 1RA1LC_0RC0RA
        'last == "2+" and rightmost_odd_idx > 0',
        # 1RB1LC_1RC---_0LD1RF_0LE1LE_0LA1LE_0RC0RB
        """last != "0"  and rightmost_odd_idx_alt > 0 and `last %2` == 1""",
        # 'last != "0" and all_even == 0 and rightmost_odd_idx != 0 and `last %2` == 1 and gray != "0" and `gray-adic` != "2^L - 1"',
        'gray == "0" and all_even == 0',
        # D1RE 0LAL1D inspired
        'last == "2+" and `last %2` == 0',
        'last == "2+" and `last %2` == 1 and gray == "0"',
        'last == "2+" and `last %2` == 1 and gray != "0"',
        "`gray-adic` == '2^L - 1' and L > 1",
        "`gray-adic` == '2^L - 1' and L == 1",
        "last == '1' and `gray-adic` != '2^L - 1' and gray != '0'",
        "last == '1' and `gray-adic` != '2^L - 1' and gray == '0' and L == 2",
        "last == '1' and `gray-adic` != '2^L - 1' and gray == '0' and L != 2",
        # 1RB---_0LC1RF_1RA1LD_0LE1RF_0LC1LE_0RD0RA
        "rightmost_odd_idx_alt == 0 and last == '1'",
        # halting stuff
        # 1RB---_0LC1RF_0LD1LC_1RE1LB_1RD0RB_0RB0RA (is there a condition on first?)
        'all_even == 1 and internal_zeros_len > 0 and last != "0"',
        # 1RB---_0RC1RF_0LD1RF_0LE1LD_1RA1LC_0RC0RA 
        'all_even == 1 and internal_zeros_len > 0 and last != "0"',
        # 1RB1LC_1RC---_0LD1RF_0LE1LE_0LA1LE_0RC0RB
        'all_even == 1 and internal_zeros_len > 0 and last == "0"',
        # 1RB1LC_1RC---_0LD1RE_0LA1LD_0RC1RF_0LF0RB
        'all_even == 1 and internal_zeros_len >= 2',
        #1RB---_0LC1LB_1RE1LD_0LB1RF_1RD1RA_0RD0RE
        #???
        # 1RB1LC_1RC---_0LD1RE_0LA1LD_0RF0RB_0LC1RE
        # no halting encountered
]

# for one_tm_string in UNIQ_TMS.split("\n"):
#     df = dfs[one_tm_string]
#     print(df.query("not (" + " or ".join(f"({c})" for c in ALL_CONDS) + ")").shape)
from dekaheptoid_beaver.list_actions import (
    apply_increment, apply_unsafe_increment_1N, apply_halt, 
    apply_overflow, apply_zero_variant, apply_zero_increment,
    apply_init, apply_halve,
    apply_zero, apply_shallow_increment, apply_excluding_increment, apply_increment_D_even,
    apply_zero_D, apply_weird_double_zero, apply_nan_safe_increment, apply_weird_overflow, apply_halve_and_increment,
    apply_weird_one_zero, apply_leading_zero_increment,
    apply_smudge_internal_zeros, apply_weird_incrementing_halve, apply_reverse_smudge_internal_zeros,
    apply_weird_recursive_increment, apply_weird_halve_zero, apply_weird_halve_zero_variant
)

ALL_RULES = [
    apply_increment, apply_unsafe_increment_1N, apply_halt, 
    apply_overflow, apply_zero_variant, apply_zero_increment,
    apply_init, apply_halve, 
    apply_zero, apply_shallow_increment, apply_excluding_increment, apply_increment_D_even,
    apply_zero_D, apply_weird_double_zero, apply_nan_safe_increment, apply_weird_overflow, apply_halve_and_increment,
    apply_weird_one_zero, apply_leading_zero_increment,
    apply_smudge_internal_zeros, apply_weird_incrementing_halve, apply_reverse_smudge_internal_zeros,
    apply_weird_recursive_increment, apply_weird_halve_zero, apply_weird_halve_zero_variant
]



with open("simulator_params.json", "r") as f:
    simulator_params = json.load(f)

import itertools

def create_2step_function(func2, func1):
    def function_template(x):
        return func2(func1(x))
    function_template.__name__ = " o ".join(f.__name__ for f in [func2, func1])
    return function_template

def create_3step_function(func3, func2, func1):
    def function_template(x):
        return func3(func2(func1(x)))
    function_template.__name__ = " o ".join(f.__name__ for f in [func3, func2, func1])
    return function_template

def create_4step_function(func4, func3, func2, func1):
    def function_template(x):
        return func4(func3(func2(func1(x))))
    function_template.__name__ = " o ".join(f.__name__ for f in [func4, func3, func2, func1])
    return function_template

ALL_RULES_LVL2 = []
for func1, func2 in itertools.product(ALL_RULES, repeat=2):
    if func1 == func2: continue
    if apply_halt in [func1, func2]: continue
    if apply_init in [func1, func2]: continue
    # new_func = lambda x: func2(func1(x))
    new_func = create_2step_function(func2, func1)
    new_func.__name__ = f"{func2.__name__} o {func1.__name__}"
    ALL_RULES_LVL2.append(new_func)

new_func = create_3step_function(apply_halve, apply_zero_variant, apply_unsafe_increment_1N)
#n ew_func.__name__ = f"{apply_halve.__name__} o {apply_zero_variant.__name__} o {apply_unsafe_increment_1N.__name__}"
ALL_RULES_LVL2.append(new_func)

new_func = create_3step_function(apply_halve, apply_zero, apply_overflow)
# new_func.__name__ = f"{apply_halve.__name__} o {apply_zero.__name__} o {apply_overflow.__name__}"
ALL_RULES_LVL2.append(new_func)

new_func = create_3step_function(apply_halve, apply_increment, apply_excluding_increment)
# new_func.__name__ = f"{apply_halve.__name__} o {apply_increment.__name__} o {apply_increment_D_odd.__name__}"
ALL_RULES_LVL2.append(new_func)

new_func = create_3step_function(apply_halve, apply_leading_zero_increment, apply_zero)
ALL_RULES_LVL2.append(new_func)

ALL_RULES_LVL2.append(
    create_4step_function(
        apply_halve, apply_weird_one_zero, apply_increment_D_even, apply_weird_double_zero
        )
)


def attempt_to_desribe_transformation(old_config, thelist):
        local_matches = []
        if len(old_config) > 1 and old_config[-1] > 1 and any(x%2 for x in old_config[:-1]):
            for apply_func in INCREMENTS_FUNCS:
                try_next = apply_func(old_config)
                if try_next == thelist:
                    return [apply_func.__name__]
        for apply_func in ALL_RULES + ALL_RULES_LVL2:
            try:
                try_next = apply_func(old_config)
            except (StopIteration, IndexError):
                try_next = []
            if try_next == thelist:
                local_matches.append(apply_func.__name__)
        return local_matches

def select_best_action(local_matches, tm_name):
    local_matches_pretty = local_matches
    
    if tm_name == '1RB---_0LC1RF_1RA1LD_0LE1RF_0LC1LE_0RD0RA':
        if 'apply_halve_and_increment o apply_excluding_increment' in local_matches:
            return 'apply_halve_and_increment o apply_excluding_increment'

    if len(local_matches) > 1:
            for priority_rule in [
                    'apply_init', 'apply_increment', 'apply_zero', 'apply_zero_variant', 
                    'apply_overflow o apply_zero_increment',
                    'apply_halve o apply_increment', 'apply_halve o apply_zero', 
                    'apply_zero_increment', 'apply_excluding_increment', 'apply_shallow_increment', 
                    # 'apply_zero_D' == apply_unsafe_increment_1N o apply_overflow
                    'apply_zero_D', 'apply_halve o apply_zero_D',
                    'apply_halve_and_increment o apply_unsafe_increment_1N', 'apply_halve o apply_weird_overflow',
                    'apply_halve o apply_weird_double_zero', 'apply_halve_and_increment o apply_increment'
                ]:
                if priority_rule in local_matches:
                    return priority_rule
            return local_matches[0]



def generate_config(N, L, n_tries=100, seed=42):
    np.random.seed(seed)
    test_configs = []
    for a_try in range(n_tries):
        config_base = np.random.randint(low=0, high=N, size=(L+2))
        config_base *= 2
        for index_type in [(-1, -1), (0, 0), (1, 1)]:
            if index_type == (-1, -1):
                config = config_base.tolist()
                config[-1] += 2
            elif index_type == (0, 0):
                config = config_base.tolist()
                config[-1] += 2
                config[0] += 1
            else:
                config = config_base.tolist()
        test_configs.append(config)
    return test_configs

def create_df_with_features(features):
    dummy = pd.DataFrame.from_dict([features])
    for c in [
            'num_leading_zeros', 'rightmost_odd_idx', 'rightmost_odd_idx_alt', 
            'L', "last %2", "L %2", 'gray %2', 'internal_zeros_len', 'total_sum', 'num_nans'
        ]:
        dummy[c] = dummy[c].astype(int)
    dummy['total_sum_mod_12'] = dummy['total_sum'] % 12

    df['all_actions'] = df['all_actions'].astype(str)
    df = df.drop(columns=['steps_taken', 'L', 'total_sum']).drop_duplicates()
    
    return dummy

def data2feature_map(entry):
    one_dict = {"all_even": 0, "all_zeros": 0, "num_leading_zeros": 0, "rightmost_odd_idx": "", "L": -1, "gray": "", "gray-adic": ""}
    for feat in entry[1]:
            # if "rightmost_odd_idx" in feat: continue
            if feat in ['all even', 'all zeros']:
                one_dict[feat.replace(" ", "_")] = 1
            if "==" in feat:
                k, _, v = feat.partition(" == ")
                if v[0] == 'f': v = v[1:]
                one_dict[k] = v
    one_dict["action"] = entry[2]
    one_dict["all_actions"] = entry[3]
    one_dict["steps_taken"] = entry[4]
    return one_dict

def list2tape(thelist, state, side):
    if side == "l":
        raise NotImplementedError
    result = "1".join(" 10^" + str(entry) +  " " for entry in thelist)
    result = result.replace("10^0", "")
    return result + state + ">"


if __name__ == "__main__":
    memoized_map_bad = {}
    data = {}
    non_increment_configs = {}
    for one_tm_string in UNIQ_TMS.split("\n"):
        if one_tm_string not in tape_evolution:
            continue

        print(one_tm_string)
        old_config = []
        old_step = 0
        memoized_map_bad[one_tm_string] = []
        non_increment_configs[one_tm_string] = []
        data[one_tm_string] = []
        for i, (step, thelist) in enumerate(tape_evolution[one_tm_string]):
            local_matches = attempt_to_desribe_transformation(old_config, thelist)
            local_matches_pretty = select_best_action(local_matches, one_tm_string)
            if len(local_matches):
                if len(old_config):
                    data[one_tm_string].append((i, list(describe_list(old_config)), local_matches_pretty, local_matches, step-old_step))
            else:
                memoized_map_bad[one_tm_string].append([old_config, thelist, i, old_step, step])    
            old_config = thelist
            old_step = step
            if i not in increment_indices[one_tm_string]:
                non_increment_configs[one_tm_string].append((i, thelist, local_matches_pretty))
        if not memoized_map_bad[one_tm_string]:
            print("All transformations are found!")
        else:
            print(len(memoized_map_bad[one_tm_string]), "unexplained transformations")
            if len(memoized_map_bad[one_tm_string]) < 20:
                print(memoized_map_bad[one_tm_string])
        print(len({"".join(x[1]) for x in data[one_tm_string]}))
        print(len(non_increment_configs[one_tm_string]), "vs", len(increment_indices[one_tm_string]))

    from collections import defaultdict
    import numpy as np



    for one_tm_string in UNIQ_TMS.split("\n"):
        print(one_tm_string)
        side, state = simulator_params[one_tm_string]
        all_features_map = []
        for config in generate_config(4, 3, seed=1) + generate_config(4, 4, seed=1) + generate_config(4, 2, seed=1) + generate_config(4, 5, seed=1):
            try:
                tm_config = list2tape(config, state, side)
                datum = [0] + [list(describe_list(config))] + ["", [], 0]
                all_features_map.append(data2feature_map(datum))
                simulated = run_direct_sim(one_tm_string, 2, start_config=tm_config, give_up_limit=4096, selected_state=state)
            except (StopIteration, IndexError):
                weird_cases[one_tm_string].append((config, None, None))
                continue

            if len(simulated) == 2:
                actual_result = tape2list(simulated[1][1])
            else:
                actual_result = "HALT"

            local_matches = attempt_to_desribe_transformation(config, actual_result)
            local_matches_pretty = select_best_action(local_matches, one_tm_string)
            if len(local_matches):
                    ok_cases[one_tm_string].append((0, list(describe_list(config)), local_matches_pretty, local_matches, 0))
            else:
                weird_cases[one_tm_string].append([config, actual_result])
        if not weird_cases[one_tm_string]:
            print("All transformations are found!")
        else:
            print(len(weird_cases[one_tm_string]), "unexplained transformations")
            if len(weird_cases[one_tm_string]) < 20:
                print(weird_cases[one_tm_string])
        print(len(ok_cases[one_tm_string]))
        # print(len({"".join(x[1]) for x in data[one_tm_string]}))

    import pandas as pd

    dfs = {}

    for one_tm_string in UNIQ_TMS.split("\n"):
        if one_tm_string not in data:
            print(one_tm_string, " not in data, WTF")
            print(one_tm_string in tape_evolution)
            continue

        df_data = []
        for entry in data[one_tm_string]:
            one_dict = data2feature_map(entry)
            df_data.append(one_dict)
        for entry in ok_cases[one_tm_string]:
            one_dict = data2feature_map(entry)
            df_data.append(one_dict)
        df = pd.DataFrame.from_dict(
            df_data
        )
        # columns=['L', 'L %2', 'gray', 'gray-adic', 'gray %2', 'last', 'last %2', 'num_leading_zeros', 'all_even', 'all_zeros', 'rightmost_odd_idx', 'action'], 

        print(df.shape)
        df['all_actions'] = df['all_actions'].astype(str)
        df['total_sum_mod_2'] = df['total_sum'] % 2
        df['total_sum_mod_3'] = df['total_sum'] % 3
        df['total_sum_mod_4'] = df['total_sum'] % 4
        df = df.drop(columns=['steps_taken', 'L', 'total_sum']).drop_duplicates()
        print(df.shape)
        df = df.fillna("")
        for c in ['num_leading_zeros', 'rightmost_odd_idx', 'rightmost_odd_idx_alt', 'L', "last %2", "L %2", 'gray %2', 'total_sum']:
            df[c] = df[c].astype(int)
        dfs[one_tm_string] = df


    import numpy as np

    rules_struct = {}

    for i1, one_tm_string in enumerate(UNIQ_TMS.split("\n")):
        df = dfs[one_tm_string]
        rules_struct[one_tm_string] = {}
        print(one_tm_string, len(df.action.unique()))
        # print(df.action.unique())
        described = {}
        for i2, Q in enumerate(ALL_CONDS):
            # print(Q)
            thelen = len(df.query(Q).action.unique())
            m[i1, i2] = thelen
            if thelen == 1: 
                pred = df.query(Q).action.unique().tolist()[0]
                if df.query(Q).index.tolist() == df.query('action == @pred').index.tolist():
                    described[Q] = pred
            if thelen > 1:
                if False and (
                    'apply_excluding_increment' in df.query(Q).action.unique() or
                    'apply_zero' in df.query(Q).action.unique()
                ):
                    pass
                    print(Q)
                    print(df.query(Q).action.value_counts())
        print(set(df.action.unique()) - set(described.values()))
        print("--------")
        for k, v in described.items():
            print(k, ':\n', v)
            rules_count[k] += 1
            rules_struct[one_tm_string][k] = v
        print()
        print()

    rules_count
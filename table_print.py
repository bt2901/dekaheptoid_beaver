
def f(thelist, i):
    while len(thelist) < 6:
        thelist = ["-"] + thelist
    digit = thelist[i]
    if digit == "-":
        return 'style="background:#111"|{{mono|-}}'
    if digit == -1:
        return 'style="background:#F22"|{{mono|-1}}'
    if digit % 2 == 0:
        return 'style="background:#0FF"|{{mono|' + str(digit) + '}}'
    else:
        return 'style="background:#DDD"|{{mono|' + str(digit) + '}}'
        
def state_s(thelist):
    parity = sum(thelist) % 2
    return -1 if parity == 0 else "+1"

print("""{| class="wikitable floatright" style="text-align:center;"
! rowspan="2"|
|-
! 5 !! 4 !! 3 !! 2 !! 1 !! 0 !! value !! length !! <math>\sigma</math>""")

for i, thelist in configs:
    thestring = f"""|-
    | {i} || {f(thelist, 0)} || {f(thelist, 1)} || {f(thelist, 2)} || {f(thelist, 3)} || {f(thelist, 4)} || {f(thelist, 5)} || {cnt2gray(thelist)}  || {len(thelist)}  || {state_s(thelist)}"""
    print(thestring)
    thelist = one_step(thelist)

print("|}")

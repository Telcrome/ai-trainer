"""
Demonstrates how to generate programs of simple languages using the probabilist
ic grammar module
"""
import itertools

from tqdm import tqdm

import trainer.lib as lib
import trainer.ml as ml

plus = lib.TS('plus')
magicnumber = lib.TS('magicnumber')
N = 3
numbers = [([lib.TS(str(i))], N - float(i)) for i in range(N)]
Number = lib.NTS('Number')
S = lib.NTS('S')


class SimpleGrammar(lib.Grammar):
    prod_rules = {
        S: [
            ([S, S], 0.1),
            # ([], 0.5),
            ([plus, Number, Number, Number, Number], 0.9),
        ],
        Number: [([magicnumber], 5.)] + numbers
    }


if __name__ == '__main__':
    sg = SimpleGrammar(S)
    print(sg)

    for prog in tqdm(sg.read_program()):
        ml.logger.log_str(prog)

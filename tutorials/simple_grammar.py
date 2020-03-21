"""
Demonstrates how to generate programs of simple languages using the probabilist
ic grammar module
"""
import itertools

import trainer.lib as lib

plus = lib.TS('plus')
magicnumber = lib.TS('magicnumber')
N = 5
numbers = [([lib.TS(str(i))], N - float(i)) for i in range(N)]
Number = lib.NTS('Number')
S = lib.NTS('S')


class SimpleGrammar(lib.Grammar):
    prod_rules = {
        S: [
            # ([S, S], 1.),
            ([plus, Number, Number, Number, Number], 1.),
        ],
        Number: [([magicnumber], 5.)] + numbers
    }


if __name__ == '__main__':
    sg = SimpleGrammar(S)
    print(sg)
    # print(sg.build_random_word())
    t = lib.ProgramSearchTree(sg)
    # t.expand_node()

    progs = [prog for prog in t._read_symbol(t.grammar.start_symbol)]

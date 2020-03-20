"""
Demonstrates how to generate programs of simple languages using the probabilistic grammar module
"""

import trainer.lib as lib

plus = lib.TS('plus')
zero = lib.TS('zero')
one = lib.TS('one')
Number = lib.NTS('Number')
S = lib.NTS('S')


class SimpleGrammar(lib.Grammar):
    prod_rules = {
        S: [
            ([S, S], 1.),
            ([plus, Number, Number], 1.),
        ],
        Number: [
            ([zero], 0.9),
            ([one], 0.1)
        ]
    }


if __name__ == '__main__':
    sg = SimpleGrammar(S)
    print(sg)
    # print(sg.build_random_word())
    t = lib.ProgramSearchTree(sg)
    # t.expand_node()
    for _ in range(10):
        prog = t.read_program(t.tree_root)
        # print(t.prog_to_str(prog))
        print(prog)

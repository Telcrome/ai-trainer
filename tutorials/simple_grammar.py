"""
Demonstrates how to generate programs of simple languages using the probabilistic grammar module
"""

import trainer.lib as lib

plus = lib.TS('plus')
magicnumber = lib.TS('magicnumber')
numbers = [([lib.TS(str(i))], float(i)) for i in range(30)]
Number = lib.NTS('Number')
S = lib.NTS('S')


class SimpleGrammar(lib.Grammar):
    prod_rules = {
        S: [
            ([S, S], 1.),
            ([plus, Number, Number], 1.),
        ],
        Number: [([magicnumber], 5.)] + numbers
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

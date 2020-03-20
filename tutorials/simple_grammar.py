"""
Demonstrates how to generate programs of simple languages using the probabilistic grammar module
"""

import trainer.lib as lib


class Action(lib.TS):
    name = 'a'


class S(lib.NTS):
    name = 'S'


class SimpleGrammar(lib.Grammar):
    prod_rules = {
        S: [
            ([Action, S], 0.5),
            ([Action], 0.5)
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
        print(t.prog_to_str(prog))

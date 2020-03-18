"""
Demonstrates how to generate programs of simple languages using the probabilistic grammar module
"""

import trainer.lib as lib


class Action(lib.TS):
    name = 'a'


class S(lib.NTS):
    name = 'S'


if __name__ == '__main__':
    sg = lib.Grammar(S)
    sg.add_prod_rule(S, [
        [Action, S], [Action]
    ])
    print(sg)

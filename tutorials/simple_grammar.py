"""
Demonstrates how to generate programs of simple languages using the probabilistic grammar module
"""

import trainer.lib as lib


class Action(lib.TS):

    def __repr__(self):
        return 'a'


class S(lib.NTS):

    def __repr__(self):
        return 'S'


if __name__ == '__main__':
    sg = lib.Grammar(S)
    sg.add_prod_rule(S, [
        [Action, S], [Action]
    ])

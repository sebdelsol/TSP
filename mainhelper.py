# coding: utf-8
# pylint: disable=missing-docstring, unused-import

import inspect

from log import log
from aco import ACO
from graph import Graph, GraphRND
from graphTsp import GraphTSP
from graphMap import GraphMAP

# tools
TAB = ' ' * 4
def tab(txt, end=None):
    print (TAB + txt, end=end)

def get_leaves_cls(cls):
    childs = cls.__subclasses__()
    if len(childs) == 0:
        yield cls

    for child in childs:
        for leaf in get_leaves_cls(child):
            yield leaf

def tab_list_of_str(alist, perline):
    txts = [f'[{i}]'.rjust(4) + f' = {name},' for i, name in enumerate(alist)]
    txts[-1] = txts[-1][:-1] # remove last ','

    mlen = max(len(txt) for txt in txts)
    for i in range(0, len(txts), perline):
        txt = ''.join(txt.ljust(mlen) for txt in txts[i: i + perline])
        tab (f'    # {txt}')

def to_str(val):
    return f"'{val}'" if isinstance(val, str) else val

def get_defaults(func):
    for key, val in inspect.signature(func).parameters.items():
        val = val.default
        if val is not inspect.Parameter.empty:
            yield key, to_str(val)


def print_defaults(txt, func, override = None):
    key_val = get_defaults(func)

    if override is not None:
        key_val = [(k, override.get(k, v)) for k, v in key_val]

    width = max(len(k) for k, v in key_val)
    args = (f'{k}'.ljust(width) + f' = {v}' for k, v in key_val)

    txt = TAB + txt + 'dict('
    sep = ',\n' + f'{" " * len(txt)}'
    print (txt + sep.join(args) + ')')


# main code helper
def copy_paste_helper():
    log ('TO COPY PASTE !')
    tab ('# generated with mainhelper.py')
    tab ('from mainHelper import GraphClasses, TspOpts, AcoClasses')
    print ()
    for i, gtype in enumerate(GraphClasses):
        tab (f'# [{i}] = {gtype.__name__}')
    tab ( 'GraphCls = GraphClasses[0]')
    tab ( 'gclsname = GraphCls.__name__')
    print ()
    for i, gtype in enumerate(GraphClasses):
        tab (f"{'if' if i==0 else 'elif'} gclsname == '{gtype.__name__}':")
        if gtype is GraphTSP:
            tab_list_of_str(TspOpts, 5)
        print_defaults ('    gkwargs = ', gtype.__init__, {'tspname': 'TspOpts[0]'})
        print ()
    tab ( 'graph = GraphCls(**gkwargs)')
    print ()
    tab ( 'if graph.isvalid:')
    for i, strat in enumerate(AcoClasses):
        tab (f'    # [{i}] = {strat.__name__}')
    tab ( '    acoCls = AcoClasses[2]')
    print ()
    print_defaults ('    acokwargs = ', ACO.__init__, {'nAnts': 'graph.nNodes'})
    print ()
    print_defaults ('    simkwargs = ', ACO.simulate)
    print ()
    tab ( '    aco = acoCls(graph, **acokwargs)')
    tab ( '    while aco.simulate(**simkwargs):')
    tab ( '        pass')
    tab ( '    aco.close()')
    print ()


# list of TSP, Graph and ACO to choose from
_, TspOpts = GraphTSP.getAllValidTsp()
GraphClasses = list(get_leaves_cls(Graph))
AcoClasses   = ACO.__subclasses__()

if __name__ == '__main__':
    copy_paste_helper()

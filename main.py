# coding: utf-8
# pylint: disable=missing-docstring, invalid-name, fixme

# biblio
# inspired by https://github.com/rochakgupta/aco-tsp

# http://www.ijmlc.org/vol8/710-SDM18-112.pdf
# http://www-igm.univ-mlv.fr/~lombardy/ens/JavaTTT0708/ants.pdf
# http://www.i3s.unice.fr/~crescenz/publications/travaux_etude/colonies_ants-200605-rapport.pdf
# nAnts = nNodes # http://www.ijmlc.org/vol8/710-SDM18-112.pdf

# install dependencies
# conda config --prepend channels conda-forge
# conda create -n ox --strict-channel-priority osmnx
# conda activate ox
# pip install tsplib95

# todo
# better scale
# better termination ?
# case when pos = 0 or pos = one value of the array ????
# openCL
# MinMax with alternation of localBest to globalBest
# matplotlib in html with folium
# opt2
# ATSP
# save best run parameters

# Main process with a guard fro multiprocess
if __name__ == '__main__':

    # generated with mainhelper.py
    from mainhelper import GraphClasses, TspOpts, AcoClasses

    # [0] = GraphRND
    # [1] = GraphTSP
    # [2] = GraphMAP
    GraphCls = GraphClasses[1]
    gclsname = GraphCls.__name__

    if gclsname == 'GraphRND':
        gkwargs = dict(nNodes = 50,
                       seed   = 1000,
                       xmax   = 1000,
                       ymax   = 1000)

    elif gclsname == 'GraphTSP':
        #  [0] = eil51,    [1] = berlin52, [2] = st70,     [3] = eil76,    [4] = pr76,
        #  [5] = kroA100,  [6] = kroC100,  [7] = kroD100,  [8] = rd100,    [9] = eil101,
        # [10] = lin105,  [11] = ch130,   [12] = ch150,   [13] = tsp225,  [14] = a280,
        # [15] = pcb442,  [16] = pr1002,  [17] = pr2392
        gkwargs = dict(tspname = TspOpts[0])

    elif gclsname == 'GraphMAP':
        gkwargs = dict(nNodes     = 50,
                       seed       = 1000,
                       address    = 'Nimes, France',
                       radius     = 1500,
                       ntype      = 'drive',
                       footprints = True,
                       folium     = False)

    graph = GraphCls(**gkwargs)

    if graph.isvalid:
        # [0] = Elitist
        # [1] = Rank
        # [2] = MaxMin
        acoCls = AcoClasses[2]

        acokwargs = dict(nAnts        = graph.nNodes,
                         doPlot       = True,
                         multiprocess = True,
                         nProfile     = 0)

        simkwargs = dict(nStep       = 200,
                         alpha       = 1.0,
                         beta        = 3.0,
                         rho         = 0.1,
                         tau         = 1.0,
                         q           = 0.6,
                         nAvg        = 10,
                         termination = True)

        aco = acoCls(graph, **acokwargs)
        while aco.simulate(**simkwargs):
            pass
        aco.close()

# coding: utf-8
# pylint: disable=missing-docstring, invalid-name, attribute-defined-outside-init, unsubscriptable-object

import os
import re
import pickle
import gzip
import tsplib95
import numpy as np

from log import log
from graph import GraphXY, Tour


# graph imported from TSPLIB
class GraphTSP(GraphXY):

    TSPdir = 'TSPLIB'
    TSPurl = 'http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/ALL_tsp.tar.gz'

    def __init__(self, tspname = 'eil51'):
        self.tsp, self.opt, nNodes = GraphTSP.getTSP(tspname)

        if self.tsp:
            self.tspname = tspname
            self.optTour = self.loadOptTour()
            super(GraphTSP, self).__init__(nNodes)

        else:
            self.isvalid = False

    def getNames(self):
        nameWoDigit = re.split(r'([\d]+)', self.tspname)[0]
        self.name = r'{%s_{%d}}' %(nameWoDigit.capitalize(), self.nNodes)   # for print
        self.pname = f'{self.tspname}'                                      # for matplotlib
        self.fname = self.pname                                             # for file

    def loadOptTour(self):
        if self.opt:
            toindex = dict((n, i) for i, n in enumerate(self.tsp.get_nodes()))
            path = [toindex[n] for n in self.opt.tours[0]]
            path = np.array(path + [path[0]]) # back to start

            length = self.tsp.trace_tours(self.opt.tours)[0]

            return Tour(length = length, path = path)

        return None

    def getNodeLabels(self):
        return np.fromiter(self.tsp.get_nodes(), int)

    # get all nodes Xs & Ys
    def getNodes(self):
        XY = np.array([self.tsp.get_display(n) for n in self.tsp.get_nodes()], dtype='d')
        return XY[:, 0], XY[:, 1] #Xs & Ys are the first & second column

    def getMatDist(self):
        nx = __import__('networkx')
        return nx.to_numpy_matrix(self.tsp.get_graph())

    # load a TSP file from TSPLIB
    @staticmethod
    def getTSP(name):
        tspdir = GraphTSP.TSPdir
        tspfile = os.path.join(tspdir, f'{name}.tsp.gz')
        optfile = os.path.join(tspdir, f'{name}.opt.tour.gz')

        if os.path.exists(tspfile):
            with gzip.open(tspfile, 'rt') as f:
                tsp = tsplib95.parse(f.read())

            if os.path.exists(optfile):
                with gzip.open(optfile, 'rt') as f:
                    opt = tsplib95.parse(f.read())
            else:
                opt = None

            needs = {'symmetric'    : tsp.is_symmetric(),
                     'depictable'   : tsp.is_depictable(),
                     'complete'     : tsp.is_complete(),
                     '2D euclidian' : tsp.edge_weight_type=='EUC_2D',
                     'not special'  : not tsp.is_special()}

            if all(needs.values()):
                log (f"{name} loaded {'with' if opt else 'without'} optimal tour")
                return tsp, opt, tsp.dimension

            else:
                errors = ' & '.join((f'{what}' for what, test in needs.items() if test is False))
                log (f'{name} INVALID, should be {errors}')

        else:
            log (f'{name} NOT available')

        return None, None, None

    # get a dict of valid TSPs in TSPLIB
    @staticmethod
    def getAllValidTsp(forceupdate = False):
        tspdir = GraphTSP.TSPdir

        # create tspdir & download if tspdir doesn't exist
        if GraphTSP.createTSPdir():
            GraphTSP.dlTSP()

        # file to store the 'state' of tspdir to trck its change
        tspchgfile = os.path.join(tspdir, f'{tspdir}.tspchg')

        # get a list of (filename, modified time, size) in tspdir in nonrecursive way
        newTspchg = []
        for fname in os.listdir(tspdir):
            if fname.endswith('.gz'): # only tsp files
                fname = os.path.join(tspdir, fname)
                stat = os.stat(fname)
                newTspchg.append((fname, stat.st_mtime, stat.st_size))
        newTspchg.sort()

        tspchgfile = os.path.join(tspdir, f'{tspdir}.tspchg')

        # get last state of tspdir if available
        tspchg = []
        if os.path.exists(tspchgfile):
            with open(tspchgfile, 'rb') as f:
                tspchg = pickle.load(f)

        # check for change in tspdir
        if tspchg != newTspchg:
            forceupdate = True
            with open(tspchgfile, 'wb') as f:
                pickle.dump(newTspchg, f)

        # get the valid tsp files list
        tspdictfile = os.path.join(tspdir, f'{tspdir}.tsps')

        if forceupdate or not os.path.exists(tspdictfile):
            log ('create valid TSP list')

            tsps = {}
            for fname in os.listdir(tspdir):
                if '.tsp.' in fname and fname.endswith('.gz'):
                    tspname = fname.split('.')[0]
                    tsp, opt, nodes = GraphTSP.getTSP(tspname)
                    if tsp:
                        tsps[tspname] = {'nodes': nodes, 'opt': opt is not None}

            # sort in nb of nodes order
            tsps = sorted(tsps.items(), key=lambda item: item[1]['nodes'])

            with open(tspdictfile, 'wb') as f:
                pickle.dump(tsps, f)

        else:
            log ('load valid TSP list')

            with open(tspdictfile, 'rb') as f:
                tsps = pickle.load(f)

        return tsps, list(name for name, value in tsps if value['opt'])

    # create TSPdir if it doesn't exist
    @staticmethod
    def createTSPdir():
        tspdir = GraphTSP.TSPdir

        # create tspdir if it doesn't exist
        if not os.path.exists(tspdir):
            os.makedirs(tspdir)
            return True

        return False

    # download all TSP
    @staticmethod
    def dlTSP():
        log ('download all TSP')

        requests = __import__('requests')
        tarfile = __import__('tarfile')
        BytesIO = __import__('io').BytesIO

        r = requests.get(GraphTSP.TSPurl)
        if r.status_code == 200:
            with BytesIO(r.content) as gzdata:
                with tarfile.open(fileobj=gzdata) as tar:
                    tar.extractall(GraphTSP.TSPdir)

                    tars = set(n.split('.')[0] for n in tar.getnames())
                    log (f'{len(tars)} tsps downloaded')
                    return


if __name__ == '__main__':
    GraphTSP.createTSPdir()
    GraphTSP.dlTSP()
    GraphTSP.getAllValidTsp()

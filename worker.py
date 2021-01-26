# coding: utf-8
# pylint: disable=missing-docstring, global-variable-undefined, invalid-name

from random import random
from bisect import bisect
from multiprocessing import RawArray
import numpy as np

# ctypes array shared in memory between processes
# they can be viewed as np.array of shape (n,m)

# only for the Workers processes !!
class SharedMatWorker:
    def __init__(self, sharedMatrix):
        # copy a SharedMatMain
        self.shape = sharedMatrix.shape
        self.dtype = sharedMatrix.dtype
        self.raw = sharedMatrix.raw

    def get(self):
        # we get the matrix as a view from the RawArray
        # workers don't have the original np.array because it can't be serialized
        return np.frombuffer(self.raw, self.dtype).reshape(self.shape)

# only for the Main process !!
class SharedMatMain:
    def __init__(self, shape, dtype):
        n, m = self.shape = shape
        self.dtype = dtype
        self.raw = RawArray(dtype, n*m)

        # we get the matrix as a view from the RawArray
        self.mat = np.frombuffer(self.raw, self.dtype).reshape(self.shape)

    def get(self):
        # the main process has the original mat
        return self.mat

    def update(self, mat):
        # we can update the mat in the main process only
        np.copyto(self.mat, mat, casting='no')

class Worker:

    @staticmethod
    def init(nNodes, log, sharedMatDist, sharedMatWeight, sharedPaths):
        Worker.nNodes = nNodes
        Worker.log = log

        # sharedMatDist, sharedMatWeight and sharedPaths are shared in memory by all processes
        Worker.MatDist = sharedMatDist.get()       # read only
        Worker.MatWeight = sharedMatWeight.get()   # read only
        Worker.Paths = sharedPaths.get()           # write only, rows update in parallel

    @staticmethod
    def ant_do_tour(ant, start, q):
        nNodes = Worker.nNodes

        # an array for choosing exploration or exploitation
        exploration = np.random.random(nNodes - 2) < q

        # the path to fill for this ant
        path = Worker.Paths[ant]

        # local copy of matWeight that'll be modified
        matWeight = Worker.MatWeight.copy()

        # start
        path[0] = _current = start

        # 0 weight for any edges to start
        matWeight[:, _current] = .0

        # choose a path for the complete tour, visit only once each node
        # start has been removed & do not need to choose when only one node left
        for i in range(1, nNodes - 1):

            # weights from _current node to all nodes, visited nodes have a 0 weight
            weights = matWeight[_current]

            # exploration
            if exploration[i-1]:
                # weights accumulation array
                accWeights = np.add.accumulate(weights) # faster than weights.cumsum()
                total = accWeights[-1]

                # next index is drawn according to the weights distribution with Monte Carlo
                pos = random() * total # faster than random.uniform(0, total)
                # bisect has issues when pos = total, random() is in [0, 1)
                # but rounding value might give sth in [0, total]
                # faster than accWeights.searchsorted(pos)
                # that has issues with pos = 0 but not pos = total
                _current = bisect(accWeights, pos)

            # exploitation
            else:
                # next index has the bigger weight
                _current = weights.argmax()

            # update path
            path[i] = _current

            # 0 weight for any edges to _current
            matWeight[:, _current] = .0

        # one node left to add to be visited,
        # the edge (current->last to be visted) has a weight != 0
        path[-2] = np.nonzero(matWeight[_current])[0]

        # back to the start
        path[-1] = start

        # path length rounded with 2 digits, since equivalent paths (by rotation or inversion)
        # might have slightly different lengths due to the sum imprecision
        # rows = (array 'from nodes'), columns = (array 'to nodes') of the path
        rows, cols = path[:-1], path[1:]
        return round(Worker.MatDist[rows, cols].sum(), 2)

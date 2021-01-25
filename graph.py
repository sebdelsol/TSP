# coding: utf-8
# pylint: disable=missing-docstring, invalid-name, attribute-defined-outside-init, no-member


import math
import os
import pickle
from decimal import Decimal
import numpy as np
from matplotlib import pyplot as plt

from log import log
from worker import SharedMatMain

# get mantissa & exponent of a number
def getManExp(number):
    d = Decimal(number)
    _, digits, exponent = d.as_tuple()
    exp = len(digits) + exponent - 1
    man = d.scaleb(-exp).normalize()
    return man, exp

# Tour object for storing tours
class Tour:

    __slots__ = ('path', 'length') # faster acces
    graph = None # shared by all tours instances

    def __init__(self, tour = None, length = np.Inf, path = None):
        if tour is None:
            self.length = length
            self.path = path
        else:
            self.length = tour.length
            self.path = tour.path.copy() # need a copy since tour.path might be changed

    def load(self, fname):
        if os.path.exists(fname):
            with open(fname, 'rb') as f:
                self.length, self.path = pickle.load(f)
                return True
        return False

    def save(self, fname):
        if self.isvalid():
            with open(fname, 'wb') as f:
                pickle.dump((self.length, self.path), f)
                return True
        return False

    def isvalid(self):
        return self.length != np.Inf

    @staticmethod
    def init_graph(graph):
        Tour.graph = graph

    def add_phero(self, delta):
        Tour.graph.add_phero_path(self.path, delta / self.length)


# symmetric graph to solve TSP
class Graph:
    isvalid = True

    bestdir = 'best'
    bestFoundTour = None
    optTour = None

    def __init__(self, nNodes):
        self.nNodes = nNodes

        # names
        self.get_names()
        log (f'{self.pname} created with {nNodes} nodes')

        # how many permutations for this TSP ?
        try:
            nPermutations = .5 * math.factorial(nNodes-1)
            self.permutations = getManExp(nPermutations)
        except OverflowError:
            self.permutations = None

        # all matrices are self.nNodes * self.nNodes and symmetrical
        # (i,j) is the edge from node i to node j
        self.matShape = (nNodes, nNodes)

        # create shared dist & weight matrices for the workers use
        self.sharedMatDist = SharedMatMain(self.matShape, 'd')
        self.matDist = self.sharedMatDist.get()

        self.sharedMatWeight = SharedMatMain(self.matShape, 'd')
        self.matWeight = self.sharedMatWeight.get()

        # get matDist
        matDist = self.get_mat_dist()

        # fill diagonal with +infinity to avoid /0 in visibility computation
        np.fill_diagonal(matDist, np.Inf)

        # update the SharedMatWeight for the workers use
        self.sharedMatDist.update(matDist)

        # best found tour
        self.load_best_found()

    # init pheros on the whole graph
    def init_phero(self, phero):
        self.matPhero = np.full(self.matShape, phero) # diagonal is not used

    # init visibility on the whole graph
    def init_visibility(self, beta):
        self.matVisib = (1. / self.matDist) # avoid \0 by having matDist diagonal filled with np.Inf
        self.matVisib **= beta

    # compute weight for each edge
    def compute_weights(self, alpha):
        matWeight = self.matPhero ** alpha
        matWeight *= self.matVisib

        # update the SharedMatWeight for the workers use
        self.sharedMatWeight.update(matWeight)

    # evaporate phero on the whole graph
    def evaporate(self, evaporation):
        self.matPhero *= evaporation

    # clamp phero on the whole graph
    def clamp(self, minPhero, maxPhero):
        self.matPhero.clip(minPhero, maxPhero, out = self.matPhero)

    # add phero on the edges of a path
    def add_phero_path(self, path, delta):
        rows, cols = path[:-1], path[1:]
        self.matPhero[rows, cols] += delta

        # don't forget it's symmetrical !
        self.matPhero[cols, rows] += delta

    # don't need to save if already an optimal tour
    def load_best_found(self):
        if not os.path.exists(self.bestdir):
            os.makedirs(self.bestdir)

        if self.optTour is None:
            self.bestFoundTour = Tour()

            self.bestfile = os.path.join(self.bestdir, self.fname + '.best')
            if self.bestFoundTour.load(self.bestfile):
                log (f'best found tour so far is {self.bestFoundTour.length}km')

    # don't need to save if already an optimal tour
    def save_best_found(self, found):
        if self.optTour is None:
            if found.length < self.bestFoundTour.length:
                self.bestFoundTour = Tour(found)

                if self.bestFoundTour.save(self.bestfile):
                    log (f'best found tour is now {found.length}km')


# graph with matplotlib regular plot using Xs, Ys and nodeLabels
class GraphXY(Graph):

    def __init__(self, nNodes):
        self.nNodes = nNodes

        # nodes coords in a xmax * ymax plane
        self.Xs, self.Ys = self.get_nodes()

        # bbox and labels for plotting
        self.bbox = (self.Xs.min(), self.Ys.min(), self.Xs.max(), self.Ys.max())
        self.nodeLabels = self.get_node_labels()

        super(GraphXY, self).__init__(nNodes)

    # plot a tour
    def plot_tour(self, tour):
        bbox = self.bbox
        ax, _ = plt.gca(), plt.gcf()

        # keep the original ratio
        ax.set_aspect('equal')

        # border around graph in %
        border = .1
        ratio = 1 / (1 + 2 * border) # to map % on [0,1] axes

        # show scale
        scale = .25 # in %
        scalecolor = '.5'
        ax.annotate('', xy=(border * ratio, 0), xycoords='axes fraction',
                    xytext=((border + scale) * ratio, 0),
                    arrowprops=dict(arrowstyle="<->", color=scalecolor, shrinkA=.0, shrinkB=.0))

        xmin, ymin, xmax, ymax = bbox
        mult = {'m': 1000, 'km': 1}
        w, h = xmax - xmin, ymax - ymin

        xscale, yscale = w * scale, h * scale

        unit = 'm' if xscale < 1. else 'km'
        ax.annotate(f'{xscale * mult[unit]} {unit}',
                    xy=((border + scale * .5) * ratio, 0), xycoords='axes fraction',
                    xytext=(0, -10), textcoords='offset pixels',
                    va='center', ha='center', color=scalecolor)

        # background color
        ax.set_facecolor('.98')

        # limits of the plot
        plt.xlim((xmin - border * w, xmax + border * w))
        plt.ylim((ymin - border * h, ymax + border * h))

        # ticks step
        plt.xticks(np.arange(xmin, xmax + xscale * .5, xscale))
        plt.yticks(np.arange(ymin, ymax + yscale * .5, xscale))

        # grid & axis color
        gridcolor = '.93'
        plt.grid(color = gridcolor)
        for spine in ax.spines:
            ax.spines[spine].set_color(gridcolor)

        # no labels & ticks on axis
        ax.tick_params(axis='both', labelbottom=False, labelleft=False, bottom=False, left=False)

        #  optimal or bestFound tour available  ?
        bestTour = None
        if self.optTour:
            bestTour, bestTxt = self.optTour, 'optimal'
        elif self.bestFoundTour.isvalid():
            bestTour, bestTxt = self.bestFoundTour, r'best_{found}'

        # plot title
        title = r'$\bf{%s} TSP$, $n_{nodes}=%d$' %(self.name, self.nNodes) + '\n'

        # plot nodes edges & legends
        txt = r'$found=%gkm$'%tour.length

        # enrich if bestTour
        if bestTour:
            bestLength =  round((1. - bestTour.length / tour.length) * 100, 2)
            txt += r' i.e. $\it%s$ $\bf%+g$%%' %(bestTxt, bestLength)
            title += '\n'

        self.plot_path(tour.path, '-g.', 'blue', txt, annotateNodes = True)
        plt.title(title)

        # plot bestTour
        if bestTour:
            label = r'$%s=%gkm$'%(bestTxt, bestTour.length)
            self.plot_path(bestTour.path, '--', '.6', label, w*.01, h*.01)

        # show legend
        plt.legend(bbox_to_anchor=(0.5, 1), loc='lower center', borderaxespad=0.,
                   ncol = 1 if bestTour else 1, frameon = False, labelspacing= 0)

    # plot all edges of a path
    def plot_path(self, path, linestyle, color, label, dx = 0, dy = 0, annotateNodes = False):
        # plot nodes edges
        Xpath, Ypath = self.Xs[path], self.Ys[path]
        if dx > .0:
            Xpath += dx
        if dy > .0:
            Ypath += dy

        plt.plot(Xpath, Ypath, linestyle, linewidth = .5, color = color, label = label)

        if annotateNodes:
            for p, x, y in zip(path, Xpath, Ypath):
                plt.annotate(f'{self.nodeLabels[p]}', (x, y),
                             xytext = (2, 0), textcoords = 'offset pixels',
                             size = 8, color = 'cornflowerblue')


# graph random generated
class GraphRND(GraphXY):

    def __init__(self, nNodes = 50, seed = 1000, xmax = 1000, ymax = 1000):
        self.seed = seed
        self.xmax, self.ymax = xmax, ymax
        super(GraphRND, self).__init__(nNodes)

    def get_names(self):
        self.pname = f'RndSeed_{self.seed}_{self.nNodes}'           # for print
        self.name = r'RndSeed_{%d, %d}'%(self.seed, self.nNodes)    # for matplotlib
        self.fname = self.pname + f'_({self.xmax}, {self.ymax})'    # for file

    def get_node_labels(self):
        return np.array([i+1 for i in range(self.nNodes)])

    # get random nodes Xs & Ys in a xmax * ymax plane
    def get_nodes(self):
        # same seed gives the same graph
        np.random.seed(self.seed)

        if self.xmax == self.ymax:
            XY = np.random.uniform(0, self.xmax, 2 * self.nNodes)
            return XY[::2], XY[1::2]

        X = np.random.uniform(0, self.xmax, self.nNodes)
        Y = np.random.uniform(0, self.ymax, self.nNodes)
        return  X, Y

    # get mat distance on the whole graph
    def get_mat_dist(self):
        # vector of complex nodes coords reshaped as an 1 * nNodes matrix so we can transpose it
        coords = (self.Xs + self.Ys * 1j)[np.newaxis]

        # np.abs is the euclidian norm on complex
        return np.abs(coords.T - coords)

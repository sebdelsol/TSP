# coding: utf-8
# pylint: disable=missing-docstring, invalid-name, attribute-defined-outside-init

import random

from multiprocessing import Pool
from multiprocessing.dummy import Pool as DummyPool
from matplotlib import pyplot as plt
from matplotlib import gridspec
import numpy as np

from log import log
from graph import Tour
from worker import Worker, SharedMatMain, SharedMatWorker

# Ant Colony Optimization class for simulation
class ACO:

    name = 'ACO'

    def __init__(self, graph, nAnts=10, opt2=False, doPlot=True, multiprocess=True, nProfile=0):
        # reset seed to have 'true' random
        random.seed()
        np.random.seed()

        self.graph = graph
        self.nAnts = nAnts

        nNodes = graph.nNodes

        # create shared paths for the workers use
        sharedPaths = SharedMatMain((nAnts, (nNodes + 1)), 'i')
        self.paths = sharedPaths.get()

        # starts is a list of nAnts nodes indexes in [0, nNodes)
        # shuffle the list so that each node is as evenly distributed as possible
        buckets = (random.sample(range(nNodes), nNodes) for _ in range(1 + nAnts // nNodes))
        # flatten and slice nAnts elements
        self.starts = [e for sublist in buckets for e in sublist][:nAnts]

        # create an array of tours for each ant to avoid a class instance creation each step
        self.tours = [Tour() for _ in range(nAnts)]
        Tour.init_graph(graph) # class attribute of Tour

        # if not multiprocess use MonoDummyPool as a mono processor pool for debug
        createPool = Pool if multiprocess else DummyPool

        # create a pool of processes of workers
        args = (nNodes, log, opt2,
                SharedMatWorker(graph.sharedMatDist),
                SharedMatWorker(graph.sharedMatWeight),
                SharedMatWorker(sharedPaths))
        self.pool = createPool(initializer = Worker.init, initargs = args)
        nProcess = self.pool._processes

        # best chunksize to avoid process call overhead
        # we want the bigger possible remaining chunksize when nAnts % nProcess != 0
        self.chunksize = nAnts // nProcess + (1 if nAnts % nProcess > 0 else 0)

        log (f'aco {self.name} created')
        log (f'{nAnts} ants created')
        n_log_process = nProcess if multiprocess else 0
        log (f"run on {n_log_process} worker process{'es' if n_log_process > 1 else ''}")
        log (f'each process handles {self.chunksize} tours')

        # plotting & profiling
        self.nProfile = nProfile
        self.doProfile = nProfile > 0
        self.doPlot = doPlot and not self.doProfile

        if self.doPlot:
            self.init_plot()

        if self.doProfile:
            self.init_profiler()

    def close(self):
        if self.doProfile:
            self.close_profiler()

        if self.doPlot:
            plt.close()

        self.pool.close()
        self.pool.join()

    # see child classes
    def add_phero_this_step(self, bestTour, step):
        pass

    def do_step(self, step):
        # compute weights on all edges only once a step
        self.graph.compute_weights(self.alpha)

        # do a tour for each ant, use the workers poll
        lengths = self.pool.starmap(Worker.ant_do_tour, self.ants, self.chunksize)

        # update the tours with the new paths & lengths
        for tour, path, length in zip(self.tours, self.paths, lengths):
            tour.path, tour.length = path, length

        # best tour for this step,
        # copied with Tour constructor since the actual tour might be modified next step
        bestTour = Tour( min(self.tours, key = lambda tour: tour.length) )

        # for plotting
        self.bestLenghtPerStep[step] = bestTour.length

        # best global tour
        if bestTour.length < self.bestGlobalTour.length:
            self.bestGlobalTour = bestTour
            self.bestTourStep = step

        # add pheromone for this step, the actual function called depends on the ACO subclass used
        self.add_phero_this_step(bestTour, step)

        # evaporate pheromone on all edges
        self.graph.evaporate(1. - self.rho)

        # update movingAvg
        self.accBest[step] = self.accBest[max(0, step-1)] + bestTour.length
        accDiff = self.accBest[step] - self.accBest[max(0, step-self.nAvg)]
        self.movingAvg[step] = accDiff / self.nAvg

        # stagnations
        if step > self.nAvg:
            if self.movingAvg[step] - self.movingAvg[step-1] == 0:
                self.nStagnations += 1

    def simulate(self, nStep=200, alpha=1.0,
                 beta=3.0, rho=.1, tau=1.0, q=.6,
                 nAvg=10, termination=True):

        log (f'simulate {nStep} steps')

        self.nStep = nStep
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.tau = tau
        self.q = q

        # for each ant args for calling WorkerAntDoTour(i, start, q), where i is the # of the ant
        # recreate since q might change
        self.ants = [(i, start, q) for i, start in enumerate(self.starts)]

        # init phero on all edges of the graph
        self.graph.init_phero(tau)

        # init visibility on all edges of the graph
        self.graph.init_visibility(beta)

        # best tours
        self.bestGlobalTour = Tour() # tour.length = np.Inf
        self.bestLenghtPerStep = np.zeros(nStep)
        self.bestTourStep = 0

        # accumulation of best length for the moving average
        self.accBest = np.zeros(nStep)
        self.movingAvg = np.zeros(nStep)
        self.nAvg = nAvg

        # stagnations
        self.nStagnations = 0

        # simulate nStep
        for step in range(nStep):
            self.do_step(step)

            if termination and self.nStagnations >= nStep // 10 :
                log (f'termination at {step} steps')
                self.nStep = step + 1
                break
        else:
            if termination:
                log (f'done, stagnations={self.nStagnations}')

        # clean first false terms of movingAvg
        self.movingAvg[: self.nAvg] = self.movingAvg[self.nAvg + 1]

        # save best Tour
        self.graph.save_best_found(self.bestGlobalTour)

        # simulate again ?
        if self.doProfile:
            self.nProfile -= 1
            return self.nProfile > 0

        elif self.doPlot:
            return self.plot()

        return False

    def init_profiler(self):
        cProfile = __import__('cProfile')

        self.profiler = cProfile.Profile()
        log (f'profile x {self.nProfile}')
        self.profiler.enable()

    def close_profiler(self):
        self.profiler.disable()

        pstats = __import__('pstats')
        stats = pstats.Stats(self.profiler).sort_stats('cumtime')
        stats.print_stats(.5) # percent of all profiled functions

    def init_plot(self):
        # layout
        self.fig = plt.figure('ACO', figsize=(15, 5))
        self.gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.5], left=.025, right=.975, top=.85)

        # callback to catch hit keys on the figure window
        self.key = None
        self.fig.canvas.mpl_connect('key_press_event', self.press)

    # called when a key is pressed on the figure
    def press(self, event):
        self.key = event.key

    def plot(self):
        log (f'plot, found = {self.bestGlobalTour.length}km')
        plt.clf()

        # plot best path
        plt.subplot(self.gs[0])
        self.graph.plot_tour(self.bestGlobalTour)

        # plot best length per step
        plt.subplot(self.gs[1])
        self.plot_length_by_step()

        # draw & wait key
        log ('hit a key (Esc to exit)')
        plt.draw()
        plt.waitforbuttonpress()

        # Esc key means exit
        return self.key != 'escape'

    def plot_length_by_step(self):
        title = r'$\bf{%sAS}$, $Ratio_{explored} =' %self.name
        title += r'\frac{%g\cdot10^{3}\/\it{tours}}' % (self.nStep * self.nAnts * .001)

        if self.graph.permutations is not None:
            title += r'{%.0f\cdot10^{%d}\/\it{permutations}}$'%self.graph.permutations
        else:
            title += r'{too\/large\/...\/\it{permutations}}$'

        plt.title(title + '\n')

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        nStep = self.nStep

        # plot the best lenght per step
        Xs = np.arange(nStep)
        plt.xlabel('steps')
        plt.ylabel('km')

        label = r'$N_{steps}=%d, m_{ants}=%d, ' %(self.nStep, self.nAnts)
        label += r'\alpha_{phero}=%.2f, \beta_{visibility}=%.2f, ' %(self.alpha, self.beta)
        label += r'\tau_0=%.2f, '%self.tau
        label += r'\rho_{evaporation}=%.2f, q_{exploration}=%.2f$'%(self.rho, self.q)
        plt.plot(Xs, self.bestLenghtPerStep[:nStep], linewidth = .7,label = label)
        ax.tick_params(axis='both', which='major', labelsize=8)

        plt.legend(bbox_to_anchor=(0.5, 1), loc='lower center', borderaxespad=0.,
                   frameon = False, labelspacing= 0)

        # plot moving average
        nAvg = self.nAvg // 2
        plt.plot(Xs[nAvg:nStep] - nAvg, self.movingAvg[nAvg:nStep], linewidth = .25, color='green' )

        # point where the minimum has been reached at first
        x = self.bestTourStep
        y = self.bestGlobalTour.length
        plt.scatter([x], [y], s = 30)
        plt.annotate(f'{x}', (x, y), xytext = (0, -20), textcoords = 'offset pixels',
                     size = 10, va = 'bottom', ha = 'center')

        # horizontal line for the best tour
        color = 'green'
        plt.plot((0, self.nStep), (y, y), '--', linewidth = .5, color = color)
        plt.annotate(f'{y}', (0, y), xytext = (-31, 0), textcoords = 'offset pixels',
                     size = 9, va = 'center', color = color)

        # keep y / x ratio to .5
        ratio = (ax.get_ylim()[1] - ax.get_ylim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0])
        ax.set_aspect(.5/ratio)


# Elitist AS strategy
class Elitist(ACO):
    name = 'Elitist'

    def add_phero_this_step(self, bestTour, step):
        # all ants add phero on their tour
        for tour in self.tours:
            tour.add_phero(self.tau)

        # best global tour add phero
        self.bestGlobalTour.add_phero(self.tau)

# Rank AS strategy
class Rank(ACO):
    name = 'Rank'

    def add_phero_this_step(self, bestTour, step):
        tours = self.tours

        # sort tours in length order
        tours.sort(key = lambda tour : tour.length)

        # keep only a percent of ants
        keep = int(round(.75 * len(tours)))

        # the better the rank, the more the ant add phero
        for i in range(keep):
            tours[i].add_phero(self.tau * (keep - i))

# MaxMin AS strategy
class MaxMin(ACO):
    name = 'MaxMin'

    def add_phero_this_step(self, bestTour, step):
        # choose either local or global best tour depending on the completion
        completion = (step + 1.) / self.nStep
        tourChosen = bestTour if completion <= .75 else self.bestGlobalTour

        # add phero on the chosen best tour
        tourChosen.add_phero(self.tau)

        # clamp  phero on the chosen best tour
        maxPhero = self.tau / ((1. - self.rho) * tourChosen.length)
        minPhero = maxPhero / ( 2 * self.graph.nNodes) # http://www.ijmlc.org/vol8/710-SDM18-112.pdf

        self.graph.clamp(minPhero, maxPhero)

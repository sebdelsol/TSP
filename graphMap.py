# coding: utf-8
# pylint: disable=missing-docstring, invalid-name, attribute-defined-outside-init

import os
import pickle
import numpy as np
from matplotlib import pyplot as plt

from log import log
from graph import Graph

# graph imported from osmnx map
class GraphMAP(Graph):

    cachedir = 'cache'

    def __init__(self, nNodes = 50, seed = 1000, address = 'Nimes, France',
                 radius = 1500, ntype = 'drive', footprints = True, folium = False):

        # imports only when the class is intancied
        ox = self.ox = __import__('osmnx')
        ox.config(log_console=False, use_cache=True)

        self.folium = folium
        if folium:
            self.webbrowser = __import__('webbrowser')
            self.folium = __import__('folium')

        self.address = address
        self.radius = radius
        self.ntype = ntype
        self.seed = seed
        self.gdf = None

        # cycling color map for a tour
        cmap, _from, _to = 'ocean', 0, .7
        self.colors = ox.plot.get_colors(nNodes//2, cmap, _from, _to, return_hex=True)
        self.colors += ox.plot.get_colors(nNodes - nNodes//2, cmap, _to, _from, return_hex=True)

        log ('get map')

        Gfile = f'{address}, {radius*.001}km, {ntype.capitalize()}.graphml'
        Gfile = os.path.join(self.cachedir, Gfile)

        if os.path.exists(Gfile):
            self.G = ox.load_graphml(Gfile)

        else:
            self.G = ox.graph_from_address(address, dist=radius, network_type=ntype)
            # works only with plot_route_folium, not plot_graph_routes
            # self.G = self.G.to_undirected().to_directed()
            ox.save_graphml(self.G, Gfile)

        if footprints:
            log ('get footprints')
            self.gdf = ox.geometries_from_address(address, {'building': True}, dist=radius)
            #gdffile = os.path.splitext(Gfile)[0] + '.gpkg'
            #self.gdf.apply(lambda c: c.astype(str) if c.name != 'geometry' else c, axis=0)
            #self.gdf.to_file(gdffile, driver = 'GPKG')

        super(GraphMAP, self).__init__(nNodes)

    def getNames(self):
        self.pname = f'Map({self.address})' # for print
        self.name = r'Map_{%s}\/'%(self.address.replace(' ', r'\/')) # for matplotlib

        # for file
        self.fname = f'Map {self.address}, {self.radius*.001}km, '
        self.fname += f'{self.ntype.capitalize()}, {self.seed}, {self.nNodes}'

    # get mat distance
    def getMatDist(self):
        ox = self.ox

        # cache dists & routes on disk
        distfile = f'{self.address}, {self.radius*.001}km, {self.ntype.capitalize()}, '
        distfile += f'({self.seed}, {self.nNodes}).dists'
        distfile = os.path.join(self.cachedir, distfile)

        if os.path.exists(distfile):
            log ('load shortest routes')

            with open(distfile, 'rb') as f:
                matDist, self.routes, self.nodeids = pickle.load(f)

        else:
            log ('compute shortest routes')

            # same seed gives the same graph
            np.random.seed(self.seed)

            # random list of nodes id in the graph
            Gnodes = list(self.G.nodes())
            # gives nNodes *different* id in [0, len(Gnodes]]
            rnds = np.random.choice(len(Gnodes), self.nNodes, replace = False)
            self.nodeids = nodeids = [Gnodes[i] for i in rnds]

            # create the routes and the matDist
            matDist = np.zeros((self.nNodes, self.nNodes))
            self.routes = [ [0] * self.nNodes for _ in range(self.nNodes)] # nNodes * nNodes

            # fill upper triangle of matDist & routes
            weight = 'length'
            for i in range(self.nNodes):
                for j in range(i + 1, self.nNodes):
                    src, dst = nodeids[i], nodeids[j]
                    route = ox.shortest_path(self.G, src, dst, weight=weight)
                    weights = ox.utils_graph.get_route_edge_attributes(self.G, route, weight)
                    matDist[i][j] = sum(weights) * .001 # in km
                    self.routes[i][j] = self.routes[j][i] = route # symmetry !

            # symmetry
            matDist += matDist.T

            with open(distfile, 'wb') as f:
                pickle.dump((matDist, self.routes, self.nodeids), f)

        return matDist

    # plot a tour
    def plotTour(self, tour):
        ox = self.ox
        ax, _ = plt.gca(), plt.gcf()

        # keep the original ratio
        ax.set_aspect('equal')

        # plot title
        title = r'$\bf{%s} TSP$, $n_{nodes}=%d$' %(self.name, self.nNodes) + '\n'
        title += r'$_{seed=%d, radius=%gkm, type=%s}$'%(self.seed, self.radius*.001, self.ntype)
        title += '\n' + r'found=%gkm'%tour.length

        if self.bestFoundTour.isvalid():
            percent = round((1. - self.bestFoundTour.length / tour.length) * 100, 2)
            title += r' i.e. ${\itbest \bf%+g\%%}$' %percent

        plt.title(title)

        # plot footprint, graph & tour
        if self.gdf is not None:
            _, ax = ox.plot_footprints(self.gdf, ax=ax, show=False, close=False,
                                       color = 'orange', alpha=0.3)

        _, ax = ox.plot_graph(self.G, show=False, close=False, ax=ax,
                              edge_color='red', edge_linewidth=0.5, edge_alpha =.8, node_size=0)

        routes = [self.routes[i][j] for i, j in zip(tour.path[:-1], tour.path[1:])]
        _, ax = ox.plot_graph_routes(self.G, routes, route_colors=self.colors,
                                     show=False, close=False, ax=ax,
                                     orig_dest_size=80, route_linewidth=1, route_alpha=1)

        # label on nodes, in path order
        nodes = [self.G.nodes[self.nodeids[nodeid]] for nodeid in tour.path[:-1]]
        ixy = [(i+1, node['x'], node['y']) for i, node in enumerate(nodes)]
        # in graph order
        #ixy = [(i+1, node['x'], node['y']) for i, node in enumerate(nodes)]
        for i, x, y in ixy:
            ax.annotate(f'{i}', (x, y), c = 'w', size = 7, va = 'center', ha = 'center')

        if self.folium:
            self.plotFolium(tour, routes, ixy, percent)

    def plotFolium(self, tour, routes, ixy, percent):
        ox = self.ox

        gmap = ox.plot_graph_folium(self.G, popup_attribute='name', weight=.5, color = 'blue')

        for i, route in enumerate(routes):
            color = self.colors[i]
            gmap = ox.plot_route_folium(self.G, route, route_map = gmap,
                                        popup_attribute='length', fit_bounds=False,
                                        weight=4, color=color, fill_opacity=.5)

        DivIcon = self.folium.features.DivIcon
        fontsize = 8
        for i, y, x in ixy: # long & lat are inversed in folium
            color = self.colors[i-1]
            self.folium.CircleMarker((x, y), fontsize-1, popup=f'<b>{i}</b>',
                                     tooltip=f'{i}', color=color,
                                     fill_opacity=1, fill_color=color, fill=True).add_to(gmap)

            d = len(f'{i}')*fontsize//2 - 1
            icon=DivIcon(icon_size=(0, 0), icon_anchor=(d, fontsize),
                         html= f'<div style="font-size: {fontsize}pt; color : white">{i}</div>')
            self.folium.Marker((x, y), icon=icon).add_to(gmap)

        title = f'{self.pname} <b>TSP</b>, nodes={self.nNodes}'
        title1 = f'seed={self.seed}, radius={self.radius*.001}km, type={self.ntype}'
        title2 = f'found={tour.length}km'
        if self.bestFoundTour.isvalid():
            title2 += ' i.e. <b>%+g%%</b>' %percent

        html = f'<h3 align="center">{title}</h2>'
        html += f'<h4 align="center">{title1}</h4>'
        html += f'<h4 align="center">{title2}</h4>'
        gmap.get_root().html.add_child(self.folium.Element(html))

        cdir = os.path.dirname(os.path.realpath(__file__))
        tmpf = os.path.join(cdir, self.cachedir, 'folium.html')
        gmap.save(tmpf)
        self.webbrowser.open(r'file://' + tmpf)

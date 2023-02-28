import random
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations, groupby
import time
from networkx.algorithms import tree
from copy import deepcopy
from tqdm import tqdm

# You can use this function to generate a random graph with 'num_of_nodes' nodes
# and 'completeness' probability of an edge between any two nodes
# If 'directed' is True, the graph will be directed
# If 'draw' is True, the graph will be drawn
def gnp_random_connected_graph(num_of_nodes: int,
                               completeness: float,
                               directed: bool = False,
                               draw: bool = False):
    """
    Generates a random graph, similarly to an Erdős-Rényi
    graph, but enforcing that the resulting graph is conneted (in case of undirected graphs)
    """
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    edges = combinations(range(num_of_nodes), 2)
    G.add_nodes_from(range(num_of_nodes))
    for _, node_edges in groupby(edges, key = lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        if random.random() < 0.5:
            random_edge = random_edge[::-1]
        G.add_edge(*random_edge)
        for e in node_edges:
            if random.random() < completeness:
                G.add_edge(*e)
    for (u,v,w) in G.edges(data=True):
        w['weight'] = random.randint(-5, 20)
    if draw:
        plt.figure(figsize=(10,6))
        if directed:
            # draw with edge weights
            pos = nx.arf_layout(G)
            nx.draw(G,pos, node_color='lightblue',
                    with_labels=True,
                    node_size=500,
                    arrowsize=20,
                    arrows=True)
            labels = nx.get_edge_attributes(G,'weight')
            nx.draw_networkx_edge_labels(G, pos,edge_labels=labels)
        else:
            nx.draw(G, node_color='lightblue',
                with_labels=True,
                node_size=500)
    return G


def bellman_ford(G):
    '''Bellman-Ford Algorithm'''
    dict = {}
    nodes = list(G.nodes())
    edges = list(G.edges(data=True))
    s = nodes[0]
    weight_dict = {}
    '''
    create a dict for each connected vertexes 
    key is 'v1,v2' and the value is weight
    '''
    for i in edges:
        weight_dict[f'{i[0]},{i[1]}'] = i[-1]['weight']

    for v in nodes:
        try:
            dict[f'{v}'] = weight_dict[f'{s},{v}']
        except KeyError:
            pass
        dict[f'{s}'] = 0
    for k in range(1,len(nodes)-1):
        for v in nodes[1:]:
            for u in nodes:
                try:
                    arg2 = dict[f'{u}']+weight_dict[f'{u},{v}']
                    dict[f'{v}'] = min(dict[f'{v}'], arg2)
                except KeyError:
                    pass

    #check if there is cycle with negative weight
    dict1 = deepcopy(dict)
    dict2 = {}
    dict2['0'] = 0
    for v in nodes[1:]:
        for u in nodes:
            try:
                arg2 = dict[f'{u}']+weight_dict[f'{u},{v}']
                dict2[f'{v}'] = min(dict[f'{v}'], arg2)
            except KeyError:
                pass
    if dict2 != dict1:
        print('Negative cycle detected')
        return None
    for key, value in dict1.items():
        print(f'Distance from {s} to {key}: {value}')


def find_time(algorithm: str, vertixes: int, posibility: float, directed: bool):
    """find the time of algorithm"""
    NUM_OF_ITERATIONS = 2
    time_taken = 0
    for i in tqdm(range(NUM_OF_ITERATIONS)):
        G = gnp_random_connected_graph(vertixes, posibility, directed, True)
        start = time.time()
        if algorithm == 'kruskal':
            tree.minimum_spanning_tree(G, algorithm="kruskal")
        if algorithm == 'kruskal_our':
            kruskal_graph(G)
        if algorithm == 'bell':
            bellman(G, i)
        if algorithm == 'bell_our':
            bellman_ford(G)
        end = time.time()
        time_taken += end - start
    return time_taken / NUM_OF_ITERATIONS


def graph_plotting(posibility: float, directed: bool):
    """"""
    # x-coordinates of left sides of bars
    names = ['our', 'networkx.algorithms']
    left = [1, 2, 3, 4, 5, 6]
    # heights of bars
    for i in [10, 20, 50, 100, 200]:
        func1 = find_time('bell_our', i, posibility, directed)
        func2 = find_time('bell', i, posibility, directed)
        height = [func1, func2]
        plt.bar(names, height, color ='maroon', width = 0.4)
        plt.xlabel(f'algorithm, {i} nodes')
        plt.ylabel('Time')
        plt.title('Bar chart')
        plt.show()
        plt

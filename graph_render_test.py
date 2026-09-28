import numpy as np
import networkx as nx
from phart import ASCIIRenderer


def make_random_graph(seed: int) -> nx.Graph:
    rng = np.random.default_rng(seed)

    graph = nx.Graph()
    graph.add_nodes_from(range(6))

    # Start with a ring so the graph is always connected.
    for person in range(6):
        graph.add_edge(person, (person + 1) % 6)

    # Add extra random connections.
    for first in range(6):
        for second in range(first + 1, 6):
            if not graph.has_edge(first, second):
                if rng.random() < 0.25:
                    graph.add_edge(first, second)

    return graph


def render_graph(graph: nx.Graph, infected: set[int], current: int) -> str:
    labels = {}

    for person in graph.nodes:
        if person == current:
            status = "V"
        elif person in infected:
            status = "I"
        else:
            status = "H"

        labels[person] = f"{person}:{status}"

    labeled_graph = nx.relabel_nodes(graph, labels)

    renderer = ASCIIRenderer(labeled_graph)
    return renderer.render()


for seed in [0, 1, 2]:
    print(f"\n--- Graph generated with seed {seed} ---")

    graph = make_random_graph(seed)

    print(
        render_graph(
            graph,
            infected={0, 2, 5},
            current=2,
        )
    )

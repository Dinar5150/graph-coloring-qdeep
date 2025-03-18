# Copyright 2022 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import matplotlib
import networkx as nx
import dimod
from dimod import BinaryQuadraticModel
from dwave.samplers import SimulatedAnnealingSampler

try:
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib.use("agg")
    import matplotlib.pyplot as plt

def build_graph(num_nodes):
    """Build graph."""

    print("\nBuilding graph...")

    G = nx.powerlaw_cluster_graph(num_nodes, 3, 0.4)
    pos = nx.spring_layout(G)
    nx.draw(G, pos=pos, node_size=50, edgecolors='k')
    plt.savefig("original_graph.png")

    return G, pos

def build_bqm(G, num_colors):
    """Build BQM model for graph coloring.
    
    In the BQM formulation:
    - Each node-color combination is a binary variable
    - We penalize:
      1. Nodes without exactly one color
      2. Adjacent nodes with the same color
    """

    print("\nBuilding binary quadratic model...")

    # Initialize the BQM
    bqm = BinaryQuadraticModel('BINARY')
    
    # Create variables for each node-color pair
    # Format: (node, color)
    
    # Constraint: Each node must have exactly one color
    for node in G.nodes:
        # Add variables for this node (one per color)
        node_variables = [(node, color) for color in range(num_colors)]
        
        # Add constraint: exactly one color per node
        # 1. Linear terms to encourage assigning at least one color
        for v in node_variables:
            bqm.add_variable(v, -1)
            
        # 2. Quadratic terms to penalize assigning more than one color
        for i, v1 in enumerate(node_variables):
            for v2 in node_variables[i+1:]:
                bqm.add_interaction(v1, v2, 2)  # Penalize having both colors
    
    # Constraint: Adjacent nodes cannot have the same color
    for u, v in G.edges:
        for color in range(num_colors):
            # Penalize adjacent nodes having the same color
            bqm.add_interaction((u, color), (v, color), 2)
    
    return bqm

def run_simulated_annealing(bqm):
    """Solve BQM using simulated annealing."""

    print("\nRunning simulated annealing sampler...")

    # Initialize the simulated annealing solver
    sampler = SimulatedAnnealingSampler()
    
    # Solve the problem using the simulated annealing
    # Run multiple sweeps to improve solution quality
    sampleset = sampler.sample(bqm, num_reads=100)
    
    # Get the lowest-energy sample
    sample = sampleset.first.sample
    
    # Process the sample to get node colors
    node_colors = {}
    for (node, color), value in sample.items():
        if value == 1:  # If this color is selected for this node
            node_colors[node] = color
    
    # Verify solution - every node should have exactly one color
    for node in bqm.variables:
        if node[0] not in node_colors:
            print(f"Warning: Node {node[0]} has no color assigned!")
    
    return node_colors

def plot_soln(sample, pos, G):
    """Plot results and save file.
    
    Args:
        sample (dict):
            Sample containing a solution. Each key is a node and each value 
            is an int representing the node's color.

        pos (dict):
            Plotting information for graph so that same graph shape is used.
    """

    print("\nProcessing sample...")

    node_colors = [sample[i] for i in G.nodes()]
    nx.draw(G, pos=pos, node_color=node_colors, node_size=50, edgecolors='k', cmap='hsv')
    fname = 'graph_result.png'
    plt.savefig(fname)

    print("\nSaving results in {}...".format(fname))

# ------- Main program -------
if __name__ == "__main__":

    num_nodes = 50

    G, pos = build_graph(num_nodes)
    num_colors = max(d for _, d in G.degree()) + 1  # Upper bound on colors needed
    
    bqm = build_bqm(G, num_colors)

    sample = run_simulated_annealing(bqm)

    plot_soln(sample, pos, G)

    colors_used = max(sample.values()) + 1
    print("\nColors used:", colors_used, "\n")
    
    # Check if the solution is valid (no adjacent nodes with same color)
    valid = True
    for u, v in G.edges:
        if sample[u] == sample[v]:
            print(f"Invalid coloring: Nodes {u} and {v} both have color {sample[u]}")
            valid = False
    
    if valid:
        print("Solution is valid - no adjacent nodes have the same color!")

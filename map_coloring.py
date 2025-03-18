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

from collections import defaultdict
import argparse
import sys

from descartes import PolygonPatch
import shapefile
import matplotlib
import networkx as nx
from dimod import BinaryQuadraticModel
from dwave.samplers import SimulatedAnnealingSampler  # For simulated annealing

try:
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib.use("agg")
    import matplotlib.pyplot as plt

def read_in_args(args):
    """Read in user specified parameters."""

    # Set up user-specified optional arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--country", default='usa', choices=['usa', 'canada'], help='Color either USA or Canada map (default: %(default)s)')
    return parser.parse_args(args)

def get_state_info(shp_file):
    """Reads shp_file and returns state records (includes state info and 
    geometry) and each state's corresponding neighbors"""

    print("\nReading shp file...")

    sf = shapefile.Reader(shp_file, encoding='CP1252')

    state_neighbors = defaultdict(list)
    for state in sf.records():
        neighbors = state['NEIGHBORS']
        try:
            neighbors = neighbors.split(",")
        except:
            neighbors = []

        state_neighbors[state['NAME']] = neighbors

    return sf.shapeRecords(), state_neighbors

def build_graph(state_neighbors):
    """Build graph corresponding to neighbor relation."""

    print("\nBuilding graph from map...")

    G = nx.Graph()
    for key, val in state_neighbors.items():
        for nbr in val:
            G.add_edge(key, nbr)

    return G

def build_bqm(G, num_colors):
    """Build BQM model for map coloring.
    
    In the BQM formulation:
    - Each state-color combination is a binary variable
    - We penalize:
      1. States without exactly one color
      2. Adjacent states with the same color
    """

    print("\nBuilding binary quadratic model...")

    # Initialize the BQM
    bqm = BinaryQuadraticModel('BINARY')
    
    # Strength of constraints
    lagrange_one_color = 5.0  # Strength for "one color per state" constraint
    lagrange_different_colors = 5.0  # Strength for "neighboring states have different colors"
    
    # Constraint: Each state must have exactly one color
    for state in G.nodes():
        # Each state should have exactly one color
        # First, add variables for this state
        state_variables = [(state, color) for color in range(num_colors)]
        
        # Linear terms to encourage having at least one color
        for var in state_variables:
            bqm.add_variable(var, -lagrange_one_color)
        
        # Quadratic terms to penalize having more than one color
        for i, var1 in enumerate(state_variables):
            for var2 in state_variables[i+1:]:
                bqm.add_interaction(var1, var2, 2.0 * lagrange_one_color)
    
    # Constraint: Neighboring states cannot have the same color
    for state1, state2 in G.edges():
        for color in range(num_colors):
            # Penalize if both states have the same color
            bqm.add_interaction((state1, color), (state2, color), lagrange_different_colors)
    
    return bqm

def run_simulated_annealing(bqm):
    """Solve BQM using simulated annealing."""

    print("\nRunning simulated annealing sampler...")

    # Initialize the simulated annealing solver
    sampler = SimulatedAnnealingSampler()
    
    # Solve the problem using simulated annealing
    # Use multiple reads to improve solution quality
    sampleset = sampler.sample(bqm, num_reads=100)
    
    # Get the lowest-energy sample
    sample = sampleset.first.sample
    
    # Convert the sample to state colors
    # For each state, find which color variable is set to 1
    soln = {}
    for (state, color), value in sample.items():
        if value == 1:
            soln[state] = color
    
    # Check that every state has exactly one color
    for state in G.nodes():
        if state not in soln:
            print(f"Warning: {state} has no color assigned. Using color 0 as default.")
            soln[state] = 0
    
    # Verify no adjacent states have the same color
    for state1, state2 in G.edges():
        if soln[state1] == soln[state2]:
            print(f"Warning: Adjacent states {state1} and {state2} have the same color {soln[state1]}.")
    
    return soln

def plot_map(sample, state_records, colors):
    """Plot results and save map file.
    
    Args:
        sample (dict):
            Sample containing a solution. Each key is a state and each value 
            is an int representing the state's color.

        state_records (shapefile.ShapeRecords):
            Records retrieved from the problem shp file.

        colors (list):
            List of colors to use when plotting.
    """

    print("\nProcessing sample...")

    fig = plt.figure()
    ax = fig.gca()

    for record in state_records:
        state_name = record.record['NAME']
        color = colors[sample[state_name]]
        poly_geo = record.shape.__geo_interface__
        ax.add_patch(PolygonPatch(poly_geo, fc=color, alpha=0.8, lw=0))

    ax.axis('scaled')
    plt.axis('off')

    fname = "map_result.png"
    print("\nSaving results in {}...".format(fname))
    plt.savefig(fname, bbox_inches='tight', dpi=300)

# ------- Main program -------
if __name__ == "__main__":

    args = read_in_args(sys.argv[1:])

    if args.country == 'canada':
        print("\nCanada map coloring demo.")
        input_shp_file = 'shp_files/canada/canada.shp'
    else:
        print("\nUSA map coloring demo.")
        input_shp_file = 'shp_files/usa/usa.shp'

    state_records, state_neighbors = get_state_info(input_shp_file)

    G = build_graph(state_neighbors)

    colors = ['red', 'yellow', 'blue', 'green']  # Updated colors
    num_colors = 4

    bqm = build_bqm(G, num_colors)

    sample = run_simulated_annealing(bqm)

    plot_map(sample, state_records, colors)

    colors_used = max(sample.values())+1
    print("\nColors used:", colors_used, "\n")
    

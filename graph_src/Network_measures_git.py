from ts2vg import NaturalVG
import pandas as pd
import graph_tool.all
import numpy as np
import datetime
import os
###############################################################################
###############################################################################
def nx2gt(nxG):
    gtG = graph_tool.Graph(directed=nxG.is_directed())

    vprop_map = {}
    for node in nxG.nodes(data=False):
        vprop_map[node] = gtG.add_vertex()

    gtG.edge_properties["weight"]  = gtG.new_edge_property("float")

    for u, v, data in nxG.edges(data=True):
        e = gtG.add_edge(vprop_map[u], vprop_map[v])
        gtG.edge_properties["weight"][e] = data['weight']

    return gtG

def weight_normalize(gt):
    g_copy = gt.copy()    
    w = g_copy.ep['weight']
    max_w = w.a.max()
    w.a /= max_w
    return g_copy
###############################################################################
def calc_path_metrics(g, weighted=False):
    if weighted:
        sp = graph_tool.topology.shortest_distance(g, weights=g.ep["weight"])
    else:
        sp = graph_tool.topology.shortest_distance(g) 

    num_vertices = g.num_vertices()
    dist_matrix = sp.get_2d_array(range(num_vertices)).astype(float)

    eccentricities = {}
    for v in g.vertices():
        dists = dist_matrix[int(v)]
        finite_dists = dists[np.isfinite(dists)]
        ecc = np.amax(finite_dists) if len(finite_dists) > 0 else float('inf')
        eccentricities[int(v)] = ecc

    avg_shortest_distance = dist_matrix[np.triu_indices(num_vertices, k=1)].mean()

    np.fill_diagonal(dist_matrix, np.inf)  
    with np.errstate(divide='ignore', invalid='ignore'):
        reciprocal_dist = 1.0 / dist_matrix
        np.fill_diagonal(reciprocal_dist, np.nan) 
        compactness = np.nanmean(reciprocal_dist)
    return avg_shortest_distance, compactness, eccentricities
###############################################################################
def compute_network_metrics(spectrum):

    # Normalize spectrum
    spectrum = (spectrum / np.max(spectrum)) * (len(spectrum)-1)

    # Build visibility graph
    vg = NaturalVG(weighted="distance")
    vg.build(spectrum)
    G_nx = vg.as_networkx()

    # Convert to graph-tool and normalize weights
    g = nx2gt(G_nx)
    g = weight_normalize(g)

    node_metrics = {}

    node_metrics["degree"] = np.array([v.out_degree() for v in g.vertices()])

    node_metrics["degree_weighted"] = np.array([
        sum(g.ep["weight"][e] for e in v.out_edges()) for v in g.vertices()
    ])

    c = graph_tool.centrality.closeness(g)
    c_w = graph_tool.centrality.closeness(g, weight=g.ep["weight"])

    node_metrics["closeness"] = np.array([c[v] for v in g.vertices()])
    node_metrics["closeness_weighted"] = np.array([c_w[v] for v in g.vertices()])

    b, _ = graph_tool.centrality.betweenness(g, norm=True)
    bw, _ = graph_tool.centrality.betweenness(g, weight=g.ep["weight"], norm=True)

    node_metrics["betweenness"] = np.array([b[v] for v in g.vertices()])
    node_metrics["betweenness_weighted"] = np.array([bw[v] for v in g.vertices()])

    _, e = graph_tool.centrality.eigenvector(g)
    _, ew = graph_tool.centrality.eigenvector(g, weight=g.ep["weight"])

    node_metrics["eigenvector"] = np.array([e[v] for v in g.vertices()])
    node_metrics["eigenvector_weighted"] = np.array([ew[v] for v in g.vertices()])

    cl = graph_tool.clustering.local_clustering(g)
    cl_w = graph_tool.clustering.local_clustering(g, weight=g.ep["weight"])

    node_metrics["clustering"] = np.array([cl[v] for v in g.vertices()])
    node_metrics["clustering_weighted"] = np.array([cl_w[v] for v in g.vertices()])

    avgd, comp, ecc = calc_path_metrics(g, weighted=False)
    avgd_w, comp_w, ecc_w = calc_path_metrics(g, weighted=True)

    node_metrics["eccentricity"] = np.array([ecc[int(v)] for v in g.vertices()])
    node_metrics["eccentricity_weighted"] = np.array([ecc_w[int(v)] for v in g.vertices()])

    global_metrics = {
        "clustering_global": graph_tool.clustering.global_clustering(g, weight=None)[0],
        "clustering_global_weighted": graph_tool.clustering.global_clustering(g, weight=g.ep["weight"])[0],
        "assortativity": graph_tool.correlations.assortativity(g, "total")[0],
        "assortativity_weighted": graph_tool.correlations.assortativity(
            g, "total", eweight=g.ep["weight"]
        )[0],
        "pseudo_diameter": graph_tool.topology.pseudo_diameter(g)[0],
        "pseudo_diameter_weighted": graph_tool.topology.pseudo_diameter(
            g, weights=g.ep["weight"]
        )[0],
        "average_degree": np.mean(node_metrics["degree"]),
        "average_degree_weighted": np.mean(node_metrics["degree_weighted"]),
        "average_distance": avgd,
        "average_distance_weighted": avgd_w,
        "compactness": comp,
        "compactness_weighted": comp_w,
    }
    return global_metrics, node_metrics
############################# 
def process_dataframe(df, data_name, start=400, end=2400, output_dir = 'res/VG_maps'):
    """
    Generate visibility-graph network metrics from spectral data.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table containing metadata columns plus spectral columns named as 
        integer wavelengths (350–2500 nm). Only columns within [start, end] are used.
        The bin size of the spectral columns is considered to be 1. 
    data_name : str
        Name used as a prefix for output files.
    start, end : int, optional
        Wavelength range (nm) to analyze. Defaults: 400–2400.
    output_dir : str, optional
        Directory where result CSV files are saved.

    Returns
    -------
    global_df : pandas.DataFrame
        Global visibility-graph metrics for each spectrum, with metadata preserved.
    node_dfs : dict[str, pandas.DataFrame]
        Node-level metrics (per wavelength) for each metric type, with metadata preserved.
    """
    spectral_cols = {str(i) for i in range(350, 2501)}    
    hsr_cols = [str(i) for i in range(start, end + 1)]
    info_cols = [c for c in df.columns if c not in spectral_cols]

    global_results = []
    node_results = {}

    for idx, row in df.iterrows():
        print(f'HSR profile number: {idx}')

        spectrum = row[hsr_cols].values.astype(float)
        global_m, node_m = compute_network_metrics(spectrum)

        global_m.update({col: row[col] for col in info_cols})
        global_results.append(global_m)

        for key, array in node_m.items():
            if key not in node_results:
                node_results[key] = []
            node_results[key].append(array)

    global_df = pd.DataFrame(global_results)
    
    global_df = global_df[info_cols + [c for c in global_df.columns if c not in info_cols]]

    node_dfs = {
    k: (
        pd.DataFrame(node_results[k], columns=hsr_cols)
        .assign(**{col: df[col].values for col in info_cols})[
            info_cols + hsr_cols
        ]
    )
    for k in node_results.keys()
    }
    
    
    os.makedirs(output_dir, exist_ok=True)
    
    global_df.to_csv(f'{output_dir}/{data_name}_global_metrics_{start}_{end}.csv', index=False)
    
    for metric_name, metric_df in node_dfs.items():
        filename = f'{output_dir}/{data_name}_{metric_name}_map_{start}_{end}.csv'
        metric_df.to_csv(filename, index=False)

###############################################################################
# Test
total1 = datetime.datetime.now()
bands_dict={'ANGERS':(400,2400)}   

in_dir = 'ANGERS_Data/'
out_dir = 'res/VG_maps/'
sp='ANGERS'   
start_lambda,end_lambda = bands_dict[sp]
data0=pd.read_csv(f'{in_dir}{sp}_spectral_data.csv').head(10)
process_dataframe(data0,sp,start=start_lambda,end=end_lambda) 

total2 = datetime.datetime.now()
print(f"Total time taken: {total2-total1}")
# ###############################################################################

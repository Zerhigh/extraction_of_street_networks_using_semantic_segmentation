import os
from PIL import Image
import json
import geojson
import numpy as np
import cv2
from skimage.morphology import skeletonize
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import sknw
import networkx as nx
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry import LineString
import statistics
import time
from kaggle.create_submission_selim import to_line_strings
from sklearn.metrics.pairwise import euclidean_distances
from osgeo import gdal, ogr, osr
from kaggle.apls.osmnx_funcs import simplify_graph
import shutil
from scipy import ndimage
import random

def flatten(l):
    """
    Flatten a list with sublists.
    returns a single list.
    """
    return [item for sublist in l for item in sublist]

def skeletonize_segmentations(image_path, save_submissions, save_graph, save_skeleton, save_mask, plot=False, single=False):
    """
    receives: data paths, save paths, booleans for plotting
    returns: True, saves submission file, graph file, skeletonised result file
    Applies post-processing steps to a segmentation result
    """

    iterating = os.listdir(image_path)
    #all_files_done = os.listdir(save_graph)
    skel1 = time.time()
    for image in tqdm(iterating):
        # if image != 'AOI_4_Shanghai_PS-MS_img1185_00_00.png':
        #     continue
        # isolate image name
        image_name = image[:-10]
        img = np.asarray(Image.open(f'{image_path}/{image}'))

        # convert to skeleton and graph, apply morphologicals
        linestrings, lens, final_img, final_graph = to_line_strings(mask=img, sigma=0.5, threashold=0.3, small_obj_size1=600, dilation=4, return_ske=False)  # sigma=0.5 small_obj_size=350

        # apply graph postprocessing
        final_graph = graph_postprocessing(final_graph, final_img, plot=plot)

        # save SpaceNet submission csv
        with open(f'{save_submissions}/{image_name}.csv', 'w') as file:
            file.write('ImageId,WKT_Pix,length_m,travel_time_s\n')
            for line, leng in zip(linestrings, lens):
                file.write(f'{image.split("_")[4]},"{line}",{leng},{leng/13.66}\n')

        # pickle graph
        pickle.dump(final_graph, open(f'{save_graph}/{image_name}.pickle', 'wb'))

        # save skeleton
        cv2.imwrite(f'{save_skeleton}/{image_name}.png', final_img)

        # save postprocessed image
        #cv2.imwrite(f'{save_mask}/{image[:-10]}.png', final_mask)

        if single:
            break
    skel2 = time.time()
    print(f'finished postprocessing in {round(skel2 - skel1, 2)}s')
    return

def skeltonize_masks(image_path, save_path, plot=False):
    """
        receives: data paths, save paths
        returns: True
        saves mask files as pickles
    """

    for image in tqdm(os.listdir(image_path)):
        # cerate base array to compensate graph construction in inflated array fro image operations
        base = np.zeros((1332, 1332))
        img = np.asarray(Image.open(f'{image_path}/{image}'))
        base[16:1316, 16:1316] += img
        # skeltonize and create graph
        ske = np.array(skeletonize(img), dtype="uint8") # base
        #ske[ske > 0] = 1
        graph = sknw.build_sknw(ske, multi=True)
        # pickle graph and rename mask name
        pickle.dump(graph, open(f'{save_path}/{os.path.splitext(image)[0]}.pickle', 'wb'))
    return True

def compare_GED_graphs(gp_graphs_path, gt_graphs_path, take_first_result, max_time, out_path):
    """
        receives: path to ground truth and proposal graphs, booleans for evaluation, saving path for result
        returns: True
        Calculates the GED for each graph in a directory.
    """

    t_OGED1 = time.time()
    all_results = {}
    # iterate over all graphs
    for gp_graph_name in tqdm(os.listdir(gp_graphs_path)):
        gt_graph_name = gp_graph_name # + '.pickle' # gp_graph_name.replace('_00_00', '')
        # print(f'{gt_graphs_path}/{gt_graph_name}')
        if os.path.exists(f'{gt_graphs_path}/{gt_graph_name}'):
            # access graphs
            gp_graph, gt_graph = pickle.load(open(f'{gp_graphs_path}{gp_graph_name}', 'rb')), pickle.load(open(f'{gt_graphs_path}{gt_graph_name}', 'rb'))

            # calculate first GED iteration
            iterations = 0
            for v in nx.optimize_graph_edit_distance(gp_graph, gt_graph):
                min_result = v
                iterations += 1
                if take_first_result:
                    break
        else:
            pass

        # save results
        all_results[gt_graph_name] = min_result #GED #min_result

    # determine mean and save to file
    print(sorted(all_results.values()))
    mean_GED = statistics.mean(all_results.values())
    with open(f'{out_path}/NEW_GED{str(round(mean_GED, 2)).replace(".", "_")}.json', 'w') as f:
        json.dump(all_results, f)
    t_OGED2 = time.time()
    print(f'mean Graph Edit Distance (GD): {round(mean_GED, 2)} in {round(t_OGED2 - t_OGED1, 2)}s')
    return True

def graph_to_gdfs(G, nodes=True, edges=True, node_geometry=True, fill_edge_geometry=True):
    """
        receives: graph, booleans to determine conversion
        returns: list of geodataframe of nodes
        Converts graph nodes to list of geodataframes.
        Adopted from OSMNX_funcs.
    """

    # code adapted and changed from https://github.com/gboeing/osmnx/blob/master/osmnx/save_load.py
    """
    Convert a graph into node and/or edge GeoDataFrames
    Parameters
    ----------
    G : networkx multidigraph
    nodes : bool
        if True, convert graph nodes to a GeoDataFrame and return it
    edges : bool
        if True, convert graph edges to a GeoDataFrame and return it
    node_geometry : bool
        if True, create a geometry column from node x and y data
    fill_edge_geometry : bool
        if True, fill in missing edge geometry fields using origin and
        destination nodes
    Returns
    -------
    GeoDataFrame or tuple
        gdf_nodes or gdf_edges or both as a tuple
    """
    if not (nodes or edges):
        raise ValueError('You must request nodes or edges, or both.')

    to_return = []
    if nodes:
        if len(G.nodes(data=True)) > 0:
            # access nodes and convert
            nodes, data = zip(*G.nodes(data=True))
            gdf_nodes = gpd.GeoDataFrame(list(data), index=nodes)

            # extract geometry
            if node_geometry:
                gdf_nodes['geometry'] = gdf_nodes.apply(lambda row: Point(row['o'][0], row['o'][1]), axis=1)

            # appl crs change here
            #gdf_nodes.crs = G.graph['crs']

            to_return.append(gdf_nodes)
        else:
            print('no nodes detected')

    # not used
    if edges:
        start_time = time.time()

        # create a list to hold our edges, then loop through each edge in the
        # graph
        edges = []
        for u, v, key, data in G.edges(keys=True, data=True):
            # for each edge, add key and all attributes in data dict to the
            # edge_details
            edge_details = {'u':u, 'v':v, 'key':key}
            for attr_key in data:
                edge_details[attr_key] = data[attr_key]

            # if edge doesn't already have a geometry attribute, create one now
            # if fill_edge_geometry==True
            if 'geometry' not in data:
                if fill_edge_geometry:
                    point_u = Point((G.nodes[u]['x'], G.nodes[u]['y']))
                    point_v = Point((G.nodes[v]['x'], G.nodes[v]['y']))
                    edge_details['geometry'] = LineString([point_u, point_v])
                else:
                    edge_details['geometry'] = np.nan

            edges.append(edge_details)

        # create a GeoDataFrame from the list of edges and set the CRS
        gdf_edges = gpd.GeoDataFrame(edges)
        #gdf_edges.crs = G.graph['crs']
        #gdf_edges.gdf_name = '{}_edges'.format(G.graph['name'])

        to_return.append(gdf_edges)
        # print('Created GeoDataFrame "{}" from graph in {:,.2f} seconds'.format(gdf_edges.gdf_name, time.time()-start_time))

    if len(to_return) > 1:
        return tuple(to_return)
    else:
        return to_return[0]

def clean_intersections(G, tolerance=15, dead_ends=False):
    """
        receives: graph, tolerance value for matching, boolean if dead ends should be considered
        returns: intersected centroids
        Reduces clustered nodes by a buffer operation.
        Adopted from OSMNX_funcs.
    """

    """
    Clean-up intersections comprising clusters of nodes by merging them and
    returning their centroids.
    Divided roads are represented by separate centerline edges. The intersection
    of two divided roads thus creates 4 nodes, representing where each edge
    intersects a perpendicular edge. These 4 nodes represent a single
    intersection in the real world. This function cleans them up by buffering
    their points to an arbitrary distance, merging overlapping buffers, and
    taking their centroid. For best results, the tolerance argument should be
    adjusted to approximately match street design standards in the specific
    street network.
    Parameters
    ----------
    G : networkx multidigraph
    tolerance : float
        nodes within this distance (in graph's geometry's units) will be
        dissolved into a single intersection
    dead_ends : bool
        if False, discard dead-end nodes to return only street-intersection
        points
    Returns
    ----------
    intersection_centroids : geopandas.GeoSeries
        a GeoSeries of shapely Points representing the centroids of street
        intersections
    """

    # if dead_ends is False, discard dead-end nodes to only work with edge
    # intersections
    if not dead_ends:
        if 'streets_per_node' in G.graph:
            streets_per_node = G.graph['streets_per_node']
        else:
            streets_per_node = 1    # count_streets_per_node(G)

        dead_end_nodes = [node for node, count in streets_per_node.items() if count <= 1]
        G = G.copy()
        G.remove_nodes_from(dead_end_nodes)

    # create a GeoDataFrame of nodes, buffer to passed-in distance, merge
    # overlaps
    gdf_nodes = graph_to_gdfs(G, edges=False)
    #print(gdf_nodes)
    buffered_nodes = gdf_nodes.buffer(tolerance).unary_union
    if isinstance(buffered_nodes, Polygon):
        # if only a single node results, make it iterable so we can turn it
        # int a GeoSeries
        buffered_nodes_list = [buffered_nodes]
    else:
        # get the centroids of the merged intersection polygons
        buffered_nodes_list = [polygon for polygon in buffered_nodes.geoms]

    # index mapping implemented by myself
    last_index = len(G.nodes)
    mappings = {k+last_index: [] for k in range(len(buffered_nodes_list))}

    # create dictionary mapping polygons to contained points
    for index, polygon in enumerate(buffered_nodes_list):
        for i, row in gdf_nodes.iterrows():
            if polygon.contains(row['geometry']):
                mappings[last_index+index].append(i)

    #print(mappings)

    unified_intersections = gpd.GeoSeries(buffered_nodes_list)
    intersection_centroids = unified_intersections.centroid
    return intersection_centroids, mappings

def extend_edge_to_node(graph_, edge, tolerance=45):
    """
        receives: graph, edge to extend
        returns: extended graph
        Extends to coordinates start and end of an edge to reach the assiged node, for dipslaying purposes.
    """

    # Get the coordinates of the start and end nodes of the edge
    graph = graph_.copy()
    u, v = edge
    pts = graph.edges[u, v, 0]['pts']
    first_coords = np.array(pts[0, :])
    second_coords = np.array(pts[-1, :])#np.array((graph.nodes[v]['o'][0], graph.nodes[v]['o'][1]))

    # Get the coordinates of the node to extend the edge to
    u_node_coords = np.array([graph.nodes[u]['o'][0], graph.nodes[u]['o'][1]]).reshape((1, 2))
    v_node_coords = np.array([graph.nodes[v]['o'][0], graph.nodes[v]['o'][1]]).reshape((1, 2))

    # match the nodes to the starting points of the edge
    if np.linalg.norm(abs(first_coords - u_node_coords)) < np.linalg.norm(abs(first_coords - v_node_coords)):
        # instert u node cords at beginnging an v node cords at end
        if u_node_coords not in pts:
            pts = np.vstack((u_node_coords, pts))

        if v_node_coords not in pts:
            pts = np.vstack((pts, v_node_coords))

    elif np.linalg.norm(abs(first_coords - v_node_coords)) < np.linalg.norm(abs(first_coords - u_node_coords)):
        # instert u node cords at end an v node cords at beginning
        if v_node_coords not in pts:
            pts = np.vstack((v_node_coords, pts))

        if u_node_coords not in pts:
            pts = np.vstack((pts, u_node_coords))

    """# Calculate the distance between the start and end nodes of the edge
    #edge_length = nx.shortest_path_length(graph, u, v, 'weight')
    # Calculate the distance between the start node of the edge and the node to extend to
    #node_distance = nx.shortest_path_length(graph, u, node, 'weight')
    # Calculate the ratio of the distance between the start node and the node to extend to
    # to the distance between the start and end nodes of the edge
    # handle length of 0 for edge cae graphs
    #if edge_length == 0:
    #    return graph
    #ratio = node_distance / edge_length

    # Insert the coordinates of the node into the pts array of the edge"""
    """if u == node:
        pts = np.vstack((node_coords, pts))
    elif v == node:
        pts = np.vstack((pts, node_coords))"""
    """
    else:
        new_pt_coords = np.array([u_coords[0] + ratio * (v_coords[0] - u_coords[0]),
                         u_coords[1] + ratio * (v_coords[1] - u_coords[1])]).reshape((1, 2))
        pts = np.vstack((pts, new_pt_coords))
        pts = np.vstack((pts, node_coords))"""

    # Update the edge attributes in the graph
    graph.edges[u, v, 0]['pts'] = pts
    return graph

def graph_postprocessing(G, img, plot):
    """
        receives: Graph, corresponding image, boolean if result should be plotted
        returns: postprocessed graph
        applies postprocessing procedures: intersection reduction, node contraction, node-edge connection
    """

    # remove small roundabaouts
    last_node = len(G.nodes)
    if last_node < 1:
        print('no nodes detected')
        return G
    gdf, mappings = clean_intersections(G, dead_ends=True)
    all_new_nodes = []

    for i, point in gdf.items():
        new_node = [point.x, point.y]
        all_new_nodes.append((last_node+i, {'pts': np.array([new_node], dtype='int16'), 'o': np.array(new_node)}))

    # add new nodes
    G.add_nodes_from(all_new_nodes)

    G_new = G.copy()
    for new_node, old_nodes in mappings.items():
        for old_node in old_nodes:
            G_new = nx.contracted_nodes(G_new, new_node, old_node, self_loops=False, copy=True)

    # problem: edges are in theory connected, but pixels to nodes are not drawn
    # remove edges with length 0
    #G_new.remove_edges_from([(u, v) for u, v, attr in G_new.edges(data=True) if attr['weight'] <= 30])

    G_new_2 = G_new.copy()
    for u, v in G_new_2.edges():
        if u is not v:
            G_new_2 = extend_edge_to_node(G_new_2, (u, v))

    # Graph simplification overcomplicated everything
    """
    G_new_2simpl = simplify_graph(G_new_2)
    # remove duplicate edges
    unique_edges = set()
    non_unique = list()
    for u, v, attr in G_new_2simpl.edges(data=True):
        if (u, v) not in unique_edges:
            unique_edges.add((u, v))
        else:
            non_unique.append((u, v))

    G_new_2simpl_rem = G_new_2simpl.copy()
    for u, v, attr in G_new_2simpl.edges(data=True):
        if (u, v) in non_unique and isinstance(attr['pts'], list):
            G_new_2simpl_rem.remove_edge(u, v)
            non_unique.remove((u, v))
    for u, v, attr in G_new_2simpl_rem.edges(data=True):
        if isinstance(attr['pts'], list):
            conc = np.vstack((attr['pts'][0], attr['pts'][1]))
            attr['pts'] = conc"""

    if plot:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        for (s, e) in G.edges():
            vals = flatten([[v] for v in G[s][e].values()])
            for val in vals:
                ps = val.get('pts', [])
                ax[0].plot(ps[:, 0], ps[:, 1], 'blue')
        for (s, e) in G_new.edges():
            vals = flatten([[v] for v in G_new[s][e].values()])
            for val in vals:
                ps = val.get('pts', [])
                ax[1].plot(ps[:, 0], ps[:, 1], 'blue')
        for (s, e) in G_new_2.edges():
            vals = flatten([[v] for v in G_new_2[s][e].values()])
            for val in vals:
                ps = val.get('pts', [])
                ax[2].plot(ps[:, 0], ps[:, 1], 'blue')
        ps_ = np.array([i[1]['o'] for i in G.nodes(data=True)])
        ax[0].plot(ps_[:, 0], ps_[:, 1], 'r.', markersize=4)
        ps_ = np.array([i[1]['o'] for i in G_new.nodes(data=True)])
        ax[1].plot(ps_[:, 0], ps_[:, 1], 'r.', markersize=4)
        ps_ = np.array([i[1]['o'] for i in G_new_2.nodes(data=True)])
        ax[2].plot(ps_[:, 0], ps_[:, 1], 'r.', markersize=4)

        ax[0].set_title(f'Graph before postprocessing')
        ax[1].set_title(f'Graph after node contraction')
        ax[2].set_title(f'Graph after edge extension')

        plt.savefig('D:/SHollendonner/graphics/postprocessing/before_after_closeup.png')
        plt.show()

    return G_new_2 #G_new_2simpl_rem

def compare_topology(gp_graphs_path, gt_graphs_path, node_snapping_distance, out_path):
    """
        receives: path to all ground truth graphs, path to all proposal graphs, value for the node snapping distance,
            path to save the metrics result
        returns: True, saves metrics result to a json file
        calculates the topology metric for all graphs in a directory
    """

    top1 = time.time()
    all_graphs_stats = {}
    pls_vals = []
    combined_vals = []

    # iterate over all gt graphs and search for corresponding prop graph
    for gp_graph_name in tqdm(os.listdir(gp_graphs_path)):
        gt_graph_name = gp_graph_name # + '.pickle' # gp_graph_name.replace('_00_00', '')

        if os.path.exists(f'{gt_graphs_path}/{gt_graph_name}'):
            gp_graph, gt_graph = pickle.load(open(f'{gp_graphs_path}{gp_graph_name}', 'rb')), pickle.load(open(f'{gt_graphs_path}{gt_graph_name}', 'rb'))

            # apply topology calculation to two graphs
            stat_dict = compare_two_graph_topology(Gt=gt_graph, Gp=gp_graph, node_snapping_distance=node_snapping_distance, plot=False)
            all_graphs_stats[gt_graph_name] = stat_dict

            # save topology results
            if stat_dict['mean_path_len_similarity'] is not None:
                pls_vals.append(stat_dict['mean_path_len_similarity'])
            if stat_dict['combined'] is not None:
                combined_vals.append(stat_dict['combined'])

    top2 = time.time()

    #    print(statistics.mean([stat['mean_offset'] for stat in all_graphs_stats.values()]))
    #print(f'mean of similar path length: {round(statistics.harmonic_mean(pls_vals) * 100, 2)}% in {round( top2 -top1 ,2)}s')

    print(f'{gp_graphs_path} mean of combined: {round(statistics.harmonic_mean(combined_vals) * 100, 2)}% in {round(top2 - top1, 2)}s')

    # save result to json, apply harmonic mean to penalize 0 values
    with open(f'{out_path}/harmonic_mean_statistics_pls_{str(round(statistics.harmonic_mean(pls_vals) * 100, 2)).replace(".", "_")}_comb{str(round(statistics.harmonic_mean(combined_vals) * 100, 2)).replace(".", "_")}.json', 'w') as f:
        json.dump(all_graphs_stats, f)

    return True

def compare_two_graph_topology(Gp, Gt, node_snapping_distance, plot=False):
    """
        receives: proposal graph, ground truth graph, max distance for nodes to match, boolean for plotting results
        returns: dict with the metrics values
        calculate he topology between two graphs
    """

    #compare the topology of two graphs
    stat_values = {'matched_nodes': None, 'mean_offset': None, 'mean_path_len_similarity': None, 'mean_path_similarity': None, 'combined':None}

    # extract nodes
    Gp_node_coords = [node[1]['o'] for node in Gp.nodes(data=True)]
    Gt_node_coords = [node[1]['o'] for node in Gt.nodes(data=True)]
    if len(Gp_node_coords) < 1:
        return stat_values

    # calculate all distances between all nodes
    distances = euclidean_distances(Gt_node_coords, Gp_node_coords)
    #print(distances)

    matched_nodes = dict()
    vector_lens = list()
    offset_headings = list()
    ref_vector = np.array([0, 1])

    # determine matching between gt and prop graph
    for own_index, n_gt in enumerate(distances):
        if n_gt.argmin() <= node_snapping_distance:
            nearest_node_index = n_gt.argmin()
            gt_node = list(Gt.nodes(data=True))[own_index]
            gp_node = list(Gp.nodes(data=True))[nearest_node_index]

            # calculate offset heading and length of offset
            vector = gp_node[1]['o'] - gt_node[1]['o']
            len_vector = np.linalg.norm(abs(vector))
            offset_heading = np.degrees(np.arccos(np.dot(vector, ref_vector) / len_vector))

            # save matchings
            matched_nodes[gt_node[0]] = gp_node[0]
            vector_lens.append(len_vector)
            offset_headings.append(offset_heading)

    # define values
    num_mathced = len(matched_nodes)
    num_total = len(Gt_node_coords)
    matched_ = num_mathced/num_total
    mean_offset = statistics.mean(vector_lens)

    # calculate the path len similarity
    master_path_len_similarity = list()
    master_path_similarity = list()

    # iterate over all matched nodes
    for gt_n, gp_n in matched_nodes.items():
        # construct all lengths from the source node by dijkstra
        length, path = nx.single_source_dijkstra(Gt, gt_n)

        # check for each node in the gp graph, if the path exists, and compare a single dijkstra to compare the lenghts
        path_len_similarity = list()
        path_similarity=list()

        for node, d_gt_len in length.items():
            # dont check its own node.. if it is matched, it will be 0 as well
            if node != gt_n and node in matched_nodes.keys():
                # get corresponging node number for gp graph
                corr_node = matched_nodes[node]

                # check that the path is available and not only contains itself (self loop)
                if nx.has_path(Gp, gp_n, corr_node) and gp_n != corr_node:
                    # calculate the dijsktra length
                    d_gp_len, d_gp_path = nx.single_source_dijkstra(Gp, gp_n, corr_node) #nx.dijkstra_path_length(Gp, gp_n, corr_node) #nx.single_source_dijkstra(Gp, gp_n, corr_node) #nx.dijkstra_path_length(Gp, gp_n, corr_node)

                    # applying the normalized absolute difference for length and node count
                    len_normalized_diff = 1 - np.abs(d_gt_len - d_gp_len) / np.maximum(d_gt_len, d_gp_len)
                    len_normalized_path = 1 - np.abs(len(path[node]) - len(d_gp_path)) / np.maximum(len(path[node]), len(d_gp_path))

                    path_len_similarity.append(len_normalized_diff)
                    path_similarity.append(len_normalized_path)
                    #print('hurrah', d_gt_len, d_gp_len, len_normalized_diff)
            else:
                # comparing against unmatched node
                pass

        if len(path_len_similarity) > 0:
            mean_path_len_similarity = statistics.mean(path_len_similarity)
            master_path_len_similarity.append(mean_path_len_similarity)

            mean_path_similarity = statistics.mean(path_similarity)
            master_path_similarity.append(mean_path_similarity)

            #combined = mean_path_len_similarity * 0.5 + mean_path_similarity * 0.25 + matched_ * 0.25
            #master_combined.append(combined)

    #print('matched nodes:', matched_, mean_offset, statistics.harmonic_mean(master_path_len_similarity))
    pls = None
    ps = None
    comb = None

    # calculate dict results values
    if len(master_path_len_similarity) > 0:
        pls = statistics.mean(master_path_len_similarity)
        ps = statistics.mean(master_path_similarity)

        # apply weighted formula
        combined = pls * 0.5 + ps * 0.25 + matched_ * 0.25
        #comb = combined #statistics.mean(master_combined)
        #print(statistics.mean(master_path_len_similarity), statistics.mean(master_path_similarity))

    stat_values['matched_nodes'] = matched_
    stat_values['mean_offset'] = mean_offset
    stat_values['mean_path_len_similarity'] = pls
    stat_values['mean_path_similarity'] = ps
    stat_values['combined'] = combined
    #stat_values = {'matched_nodes': matched_, 'mean_offset': mean_offset, 'harmonic_mean_path_len_similarity': pls}

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        for (s, e) in Gp.edges():
            vals = flatten([[v] for v in Gp[s][e].values()])
            for val in vals:
                ps = val.get('pts', [])
                ax.plot(ps[:, 0], ps[:, 1], 'blue')
        for (s, e) in Gt.edges():
            vals = flatten([[v] for v in Gt[s][e].values()])
            for val in vals:
                ps = val.get('pts', [])
                ax.plot(ps[:, 0], ps[:, 1], 'green')
        ps_ = np.array([i[1]['o'] for i in Gp.nodes(data=True)])
        ax.plot(ps_[:, 0], ps_[:, 1], 'r.', markersize=4)
        ps_ = np.array([i[1]['o'] for i in Gt.nodes(data=True)])
        ax.plot(ps_[:, 0], ps_[:, 1], 'r.', markersize=4)
        ps_matched = np.array([i[1]['o'] for i in Gt.nodes(data=True) if i[0] in list(matched_nodes.keys())])
        ax.scatter(ps_matched[:, 0], ps_matched[:, 1], s=20, c='black') #, 'black', markersize=6

        plt.show()

    return stat_values

def calc_F1_for_img(gt, prop):
    """
        receives: ground truth image, proposal image
        returns: F1 value for input images
        calculates F1 of two images
    """

    # calculate true pos., false pos. and false neg.
    tp = np.sum(np.logical_and(prop == 1, gt == 1))
    fp = np.sum(np.logical_and(prop == 1, gt == 0))
    fn = np.sum(np.logical_and(prop == 0, gt == 1))

    # avoid division by zero
    if tp == 0:
        tp = 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    # return F1
    return 2 * ((precision * recall)/(precision + recall))

def calc_F1_for_all(base_path, post_proc_state, gt_path):
    """
       receives: proposal path, folder direction for stitchd or postprocessed images, ground truth path
       returns: True, saves F1 value to file
       calculates F1 of all images in two directorie
   """

    # define paths
    data_path = f'{base_path}/{post_proc_state}/'

    f1_results = list()
    f1_vegas = list()
    f1_paris = list()
    f1_shanghai = list()
    f1_khartoum = list()

    for img in tqdm(os.listdir(data_path)):
        # if img != 'AOI_4_Shanghai_PS-MS_img1185_00_00.png':
        #     continue

        # read data in, aplly naming switch
        prop = read_image(f'{data_path}{img}')
        if '_0' in img:
            img = os.path.splitext(img)[0][:-6] + '.png'
        if 'MS' in img:
            img = img.replace('MS', 'RGB')
        gt = read_image(f'{gt_path}{img}')

        # if prop is not binary yet, convert it
        if list(np.unique(prop)) != [0, 1]:
            prop[prop >= 1] = 1

        # calc F1 for images
        res = calc_F1_for_img(gt=gt, prop=prop)

        # allocate F1 scores
        if 'Vegas' in img:
            f1_vegas.append(res)
        if 'Paris' in img:
            f1_paris.append(res)
        if 'Shanghai' in img:
            f1_shanghai.append(res)
        if 'Khartoum' in img:
            f1_khartoum.append(res)
        f1_results.append(res)

    print("total f1 mean:", statistics.mean(f1_results))
    print("vegas f1 mean:", statistics.mean(f1_vegas))
    print("paris f1 mean:", statistics.mean(f1_paris))
    print("shanghai f1 mean:", statistics.mean(f1_shanghai))
    print("khartoum f1 mean:", statistics.mean(f1_khartoum))

    # write results to file
    with open(f'{base_path}/F1_scores_{post_proc_state}.txt', 'w') as file:
        file.write(f"total f1 mean: {statistics.mean(f1_results)}\n")
        file.write(f"vegas f1 mean: {statistics.mean(f1_vegas)}\n")
        file.write(f"paris f1 mean: {statistics.mean(f1_paris)}\n")
        file.write(f"shanghai f1 mean: {statistics.mean(f1_shanghai)}\n")
        file.write(f"khartoum f1 mean: {statistics.mean(f1_khartoum)}\n")

    return

def read_image(source):
    image = np.asarray(Image.open(source)).flatten()
    return image

def determine_overlap(img_size, wish_size):
    """
        receives: image size to split, size image is split into
        returns: list of tuples describing the indices to split an image along
        calculates indices on whichan image has to be split
    """
    num_pics = int(np.ceil(img_size/wish_size))
    applied_step = int((num_pics * wish_size - img_size) / (num_pics - 1))
    overlap_indices = [(i*(wish_size-applied_step), (i+1)*wish_size - i*applied_step) for i in range(num_pics)]

    return overlap_indices

def stitch_overlap_images(sorted_images, result_path, overlap_params, old_segmentation_result, for_visual_output):
    """
        receives: dict of sorted image names, path where to save resuls, dict with parameters describing overlpping,
            boolean for color allocation, boolean if result needs to be inspected visually
        returns: Tue, results are saved
        stitches images back together, after being plit for segmentation
    """

    # images need to be binary
    if old_segmentation_result:
        street_color = 207
        background_color = 20
    else:
        street_color = 207
        background_color = 0
    if for_visual_output:
        output_color = 255
    else:
        output_color = 1

    for k, img_paths in tqdm(sorted_images.items()):
        # change base to 1301 size and crop later to allow overlap division problem
        base = np.zeros((1301, 1301))
        arrays = dict()
        base_name = img_paths[0].split('/')[-1]

        for img in img_paths:
            names = img.replace('.png', '').split('/')[-1].split('_')[-2:]
            ids = (int(names[0]), int(names[1]))
            image = Image.open(img)
            # rescale image to fit overlap parameter
            if image.size[0] != overlap_params[0][1]:
                image = image.resize((overlap_params[0][1], overlap_params[0][1]))

            # convert to array
            image_open = np.asarray(image)
            # check if image is single channel, if not, convert to single channel
            if len(image_open.shape) > 2 and image_open.shape[2] > 1:
                image_open = image_open[:, :, 0]

            # rescale color values to binary
            ret_image = image_open.copy()
            ret_image[ret_image < street_color] = 0
            ret_image[ret_image >= street_color] = 1
            arrays[ids] = ret_image

        # place arrays in big array
        it = 0
        for i, i_val in enumerate(overlap_params):
            for j, j_val in enumerate(overlap_params):
                if (i, j) in arrays.keys():
                    img = arrays[i, j]
                    base[i_val[0]:i_val[1], j_val[0]:j_val[1]] += img
                it += 1
        base[base > 0] = output_color # 1: for binary, 255: for visual

        # save files
        cv2.imwrite(f'{result_path}/{base_name}', base[:1300, :1300])

    return True

def sort_images(base_path):
    """
        receives: a path to results images
        returns: dict with images sorted back together
        sorts all files, to allocate split files back together
    """

    ret_dict = dict()
    # retrieve the image number from a string like this: 'AOI_2_Vegas_PS-RGB_img1_00_00.png' -> img1
    for name in tqdm(os.listdir(base_path)):
        ins_name = f'{name.split("_")[2]}_{name.split("_")[4]}'
        if ins_name in ret_dict.keys():
            ret_dict[ins_name].append(base_path+name)
        elif ins_name not in ret_dict.keys():
            ret_dict[ins_name] = [base_path+name]

    return ret_dict

def getGeom(inputRaster, sourceSR='', geomTransform='', targetSR=''):
    """
        receives: input image, source spatial reference, transformation parameter, target spatial reference
        returns: the inputs geometry
        copied from the OSM library
    """
    # from osmnx
    if targetSR == '':
        performReprojection = False
        targetSR = osr.SpatialReference()
        targetSR.ImportFromEPSG(4326)
    else:
        performReprojection = True

    if geomTransform == '':
        srcRaster = gdal.Open(inputRaster)
        geomTransform = srcRaster.GetGeoTransform()

        source_sr = osr.SpatialReference()
        source_sr.ImportFromWkt(srcRaster.GetProjectionRef())

    # geom = ogr.Geometry(ogr.wkbPoint)
    return geomTransform

def pixelToGeoCoord(xPix, yPix, geomTransform):
    """
        receives: xpixel, ypixel, geometry of the image
        retruns: transformed tuple of coordinates
        copied from the APLS metrics script
    """
    # If you want to gauruntee lon lat output, specify TargetSR  otherwise, geocoords will be in image geo reference
    # targetSR = osr.SpatialReference()
    # targetSR.ImportFromEPSG(4326)
    # Transform can be performed at the polygon level instead of pixel level

    """if targetSR == '':
        performReprojection = False
        targetSR = osr.SpatialReference()
        targetSR.ImportFromEPSG(4326)
    else:
        performReprojection = True

    if geomTransform == '':
        srcRaster = gdal.Open(inputRaster)
        geomTransform = srcRaster.GetGeoTransform()

        source_sr = osr.SpatialReference()
        source_sr.ImportFromWkt(srcRaster.GetProjectionRef())
    """

    # extract geometry
    geom = ogr.Geometry(ogr.wkbPoint)
    xOrigin = geomTransform[0]
    yOrigin = geomTransform[3]
    pixelWidth = geomTransform[1]
    pixelHeight = geomTransform[5]

    # apply coordinate transformation
    xCoord = (xPix * pixelWidth) + xOrigin
    yCoord = (yPix * pixelHeight) + yOrigin
    geom.AddPoint(xCoord, yCoord)

    """if performReprojection:
        if sourceSR == '':
            srcRaster = gdal.Open(inputRaster)
            sourceSR = osr.SpatialReference()
            sourceSR.ImportFromWkt(srcRaster.GetProjectionRef())
        coord_trans = osr.CoordinateTransformation(sourceSR, targetSR)
        geom.Transform(coord_trans)"""

    return (geom.GetX(), geom.GetY())

def plot_graph(G_p):
    """
        receives: graph
        retruns: None
        plots a graph
    """

    # draw edges by pts
    for (s, e) in G_p.edges():
        vals = flatten([[v] for v in G_p[s][e].values()])
        for val in vals:
            ps = val.get('pts', [])
            plt.plot(ps[:, 1], ps[:, 0], 'green')

    nodes = G_p.nodes(data=True)
    ps = np.array([i[1]['o'] for i in nodes])

    plt.plot(ps[:, 1], ps[:, 0], 'r.')

    plt.title('Build Graph')
    plt.show()

def convert_graph_to_geojson(G_g):
    """
        receives: graph
        retruns: point features of graph, line features of graph
        converts a graphs nodes and edges into a geojson
    """

    point_features, linestring_features = [], []

    for node in G_g.nodes(data=True):
        point = geojson.Point((node[1]['coords'][0], node[1]['coords'][1]))
        feature = geojson.Feature(geometry=point, properties={'id': node[0]})
        point_features.append(feature)

    for start, stop, attr_dict in G_g.edges(data=True):
        coords = attr_dict['coords']
        line = geojson.LineString(coords)

        feature = geojson.Feature(geometry=line, properties={})
        linestring_features.append(feature)

    return point_features, linestring_features

def convert_all_graphs_to_geojson(graph_path, RGB_image_path, MS_image_path, out_path, ms_bool):
    """
        receives: path to graphs, path to RGB images, path to MS images, saving path, boolean if input is MS or RGB
        retruns: True, saves geojsons
        converrt all graphs into georeferenced geojsons and save 3 files per graph
    """

    all_images = list()
    geosjon_time = time.time()

    # if RGB, access RG images and copy them to a single list
    if not ms_bool:
        img_type = 'PS-RGB_8bit'
    else:
        img_type = 'PS-MS'

    # extract rgb images from nested folder structure
    image_path = RGB_image_path
    for folder in os.listdir(RGB_image_path):
        for img in os.listdir(f'{RGB_image_path}/{folder}/{img_type}/'):
            all_images.append(f'{folder}/{img_type}/{img}') #SN3_roads_train_

    for img in tqdm(all_images):
        # extract images
        if not ms_bool:
            name = os.path.splitext(img)[0].split('/')[-1].replace('SN3_roads_train_', '')
        else:
            if os.path.splitext(img)[1] == '.tif':
                name = os.path.splitext(img)[0].split('/')[-1].replace('SN3_roads_train_', '') #os.path.splitext(img)[0]
            else:
                continue

        # extract graph
        if os.path.exists(f'{graph_path}{name}.pickle'):
            with open(f'{graph_path}{name}.pickle', "rb") as openfile:
                # load graph and extract geometry
                G = pickle.load(openfile)
                geom = getGeom(f'{image_path}{img}')

                # apply transformation to nodes
                for i, (n, attr_dict) in enumerate(G.nodes(data=True)):
                    x_pix, y_pix = attr_dict['pts'][0][1], attr_dict['pts'][0][0]
                    x_WGS, y_WGS = pixelToGeoCoord(x_pix, y_pix, geomTransform=geom)
                    attr_dict["coords"] = (x_WGS, y_WGS)

                # apply transformation to edges
                for start, stop, attr_dict in G.edges(data=True):
                    coords = list()
                    for point in attr_dict['pts']:
                        coords.append(pixelToGeoCoord(point[1], point[0], geomTransform=geom))
                    attr_dict['coords'] = coords

                point_features, linestring_features = convert_graph_to_geojson(G) #f'{image_path}{name}.png')

                feature_collection_points = geojson.FeatureCollection(point_features)
                feature_collection_linestrings = geojson.FeatureCollection(linestring_features)

                # Write GeoJSON to file
                with open(f'{out_path}/qgis_geojsons/{name}_points.geojson', 'w') as f:
                    geojson.dump(feature_collection_points, f)
                with open(f'{out_path}/qgis_geojsons/{name}_linestrings.geojson', 'w') as f:
                    geojson.dump(feature_collection_linestrings, f)
                with open(f'{out_path}/sub_geojsons/{name}.geojson', 'w') as f:
                    geojson.dump(geojson.FeatureCollection(point_features + linestring_features), f)

    geosjon_time2 = time.time()
    print(f'created geojsons in {round(geosjon_time2 - geosjon_time, 2)}s')
    return True

# from_path = 'C:/Users/shollend/bachelor/kaggle/output/All_190223_resnet50_e150/'
models = ['2605_512_unet_densenet201_MS_200epochs_full_continued']
"""models = ['2104_256_basic_unet_densenet201_RGB_150epochs',
          '2304_256real_basic_unet_densenet201_RGB_150epochs' ,
          'unet3plus_1804_256_basic_unet_deepnet_RGB_200epochs',
          'unet3plus_1504_256_att_unet_RGB_200epochs',
          '0205_512real_basic_unet_densenet201_RGB_150epochs',
          '0605_512_unet_densenet201_RGB_150epochs_small',
          '0705_512_unet_densenet201_RGB_150epochs_small',
          '0805_512_unet_densenet201_RGB_150epochs_small_RESULTSFORSMALLSET',
          '0905_512_unet_densenet201_RGB_150epochs_small_rotatetd',
          '1105_512_unet_attention_RGB_150epochs_small',
          '1105_512_unet_attention_RGB_150epochs_small_rotation',
          '1305_512_unet_densenet201_MS_150epochs_small',
          '1405_512_unet_densenet201_MS_150epochs_small',
          '2205_512_unet_densenet201_MS_200epochs_full',
          '2605_512_unet_densenet201_MS_200epochs_full_continued']"""
srt = time.time()

# skeltonize_masks(image_path='D:/SHollendonner/not_tiled/rehashed/', save_path='D:/SHollendonner/not_tiled/mask_graphs_RGB')

for name in models:
    base_path = f'D:/SHollendonner/segmentation_results/{name}/'
    print(name)
    from_path_gt_masks = 'D:/SHollendonner/not_tiled/mask_graphs_RGB/'
    if 'MS' in name:
        from_path_gt_masks = 'D:/SHollendonner/not_tiled/mask_graphs_MS/'  # 'D:/SHollendonner/not_tiled/mask_graphs_MS'

    from_results_path = f'D:/SHollendonner/segmentation_results/{name}/results/'
    from_gt_rehashed = 'D:/SHollendonner/not_tiled/rehashed/'
    from_path_stitched = f'D:/SHollendonner/segmentation_results/{name}/stitched/'
    to_path_stitched_postprocessed = f'D:/SHollendonner/segmentation_results/{name}/stitched_postprocessed/'
    to_path_skeletons = f'D:/SHollendonner/segmentation_results/{name}/skeletons/'
    to_path_gp_graphs = f'D:/SHollendonner/segmentation_results/{name}/graphs/'
    to_path_gp_graphs_not_PP = f'D:/SHollendonner/segmentation_results/{name}/graphs_not_postprocessed/'
    #to_path_gp_graphs = to_path_gp_graphs_not_PP
    to_path_submissions = f'D:/SHollendonner/segmentation_results/{name}/submissions/'
    to_path_geojsons = f'D:/SHollendonner/segmentation_results/{name}/geojsons/'
    from_RGB_img_root = 'D:/SHollendonner/data_3/'
    from_MS_img_root = 'D:/SHollendonner/data_3/' #'D:/SHollendonner/multispectral/channels_257/images/'

    print('creating filesystem')
    if not os.path.exists(from_path_stitched):
        os.mkdir(from_path_stitched)
    if not os.path.exists(to_path_skeletons):
        os.mkdir(to_path_skeletons)
    if not os.path.exists(to_path_stitched_postprocessed):
        os.mkdir(to_path_stitched_postprocessed)
    if not os.path.exists(to_path_gp_graphs):
        os.mkdir(to_path_gp_graphs)
    if not os.path.exists(to_path_submissions):
        os.mkdir(to_path_submissions)
    if not os.path.exists(to_path_geojsons):
        os.mkdir(to_path_geojsons)
    if not os.path.exists(f'{to_path_geojsons}/qgis_geojsons/'):
        os.mkdir(f'{to_path_geojsons}/qgis_geojsons/')
    if not os.path.exists(f'{to_path_geojsons}/sub_geojsons/'):
        os.mkdir(f'{to_path_geojsons}/sub_geojsons/')

    # all together onr big dataset ca 30 min
    # all together one small wo rotation dataset ca 30 min
    stitch = False
    skeletonize = False
    to_geojson = False
    calc_F1 = False
    calc_GED = False
    calc_topo = True

    if stitch:
        print('sorting images')
        sorted_images = sort_images(from_results_path)

        print('determine overlap')
        overlap = determine_overlap(1300, 512)

        print('stitch images')
        stitch_overlap_images(sorted_images=sorted_images,
                          result_path=from_path_stitched,
                          overlap_params=overlap,
                          old_segmentation_result=False,
                          for_visual_output=True)

    # test apls metric with ground truth comparison graph
    if skeletonize:
        print('skeletonise results, create graphs, apply postprocessing')
        skeletonize_segmentations(image_path=from_path_stitched,
                              save_submissions=to_path_submissions,
                              save_graph=to_path_gp_graphs,
                              save_skeleton=to_path_skeletons,
                              save_mask=to_path_stitched_postprocessed,
                              plot=False,
                              single=False)

    if to_geojson:
        print('converting graphs to geojsons')
        convert_all_graphs_to_geojson(graph_path=to_path_gp_graphs,
                                      RGB_image_path=from_RGB_img_root,
                                      MS_image_path=from_MS_img_root,
                                      out_path=to_path_geojsons,
                                      ms_bool=True)

    postproc_states = ['stitched', 'stitched_postprocessed']
    if calc_F1:
        print('calculate F1 score')
        calc_F1_for_all(base_path=base_path,
                        post_proc_state='stitched',
                        gt_path=from_gt_rehashed)

    if calc_GED:
        print('calculate GED score')
        compare_GED_graphs(gp_graphs_path=to_path_gp_graphs,
                           gt_graphs_path=from_path_gt_masks,
                           take_first_result=True,
                           max_time=10,
                           out_path=base_path)

    if calc_topo:
        #print('calculate similar path length score')
        compare_topology(gp_graphs_path=to_path_gp_graphs,
                         gt_graphs_path=from_path_gt_masks,
                         node_snapping_distance=30,
                         out_path=base_path)

stp = time.time()
print(f"complete computation took: {round(stp-srt, 2)}s")

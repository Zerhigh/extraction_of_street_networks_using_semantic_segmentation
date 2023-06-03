import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import glob
import pickle
import json
import cv2
from tqdm import tqdm
from PIL import Image
import random

def determine_overlap(img_size, wish_size):
    # same as in evaluate_graphs.py

    num_pics = int(np.ceil(img_size/wish_size))
    applied_step = int((num_pics * wish_size - img_size) / (num_pics - 1))
    overlap_indices = [(i*(wish_size-applied_step), (i+1)*wish_size - i*applied_step) for i in range(num_pics)]

    return overlap_indices

def flatten(l):
    return [item for sublist in l for item in sublist]

def overlap_graphic(size, show, save, path):
    # plot the overlapping areas of tiled images

    fig1, ax1 = plt.subplots(1,1, figsize=(10, 10))
    overlaps = determine_overlap(1300, size)
    plt.rcParams.update({'font.size': 50})
    base = np.zeros((1300, 1300))
    num_pics = len(overlaps)**2
    colors = ['#000000', '#1c1c1c', '#383838', '#555555', '#717171', '#8d8d8d', '#aaaaaa', '#c6c6c6', '#e3e3e3']
    cmap = mcolors.ListedColormap(colors)

    for o1 in overlaps:
        for o2 in overlaps:
            base[o1[0]:o1[1], o2[0]:o2[1]] += 1

    total = 1300**2
    ol_one = round(base[base == 1].shape[0]/total * 100, 2)
    ol_two = round(base[base == 2].shape[0]/total * 100, 2)
    ol_four = round(base[base == 4].shape[0]/total * 100, 2)

    print(ol_one, ol_two, ol_four)

    """legend_elements = [Patch(facecolor=colors[0], label=f'covered one time \n& in total {ol_one}%'),
                       Patch(facecolor=colors[2], label=f'covered two times \n& in total {ol_two}%'),
                       Patch(facecolor='#ffffff', label=f'covered four times \n& in total {ol_four}%')]"""

    #ax1.set_title(f'Displaying overlap of image tiled into {num_pics} sub-images with size {size}px')
    ax1.imshow(base, cmap=cmap)
    #ax1.legend(handles=legend_elements, loc='upper right')
    ax1.set_xlabel('image pixel', fontsize=20)
    ax1.set_ylabel('image pixel', fontsize=20)
    ax1.invert_yaxis()

    if save:
        plt.savefig(f'{path}/overlap_{size}.png')
    if show:
        plt.show()

def plot_graph(G_p, show, save, path, title, G_t=None):
    # plot a proposal (and a ground truth) graph

    if G_t is not None:
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))

        for (s, e) in G_p.edges():
            vals = flatten([[v] for v in G_p[s][e].values()])
            for val in vals:
                ps = val.get('pts', [])
                ax[0].plot(ps[:, 1], ps[:, 0], 'green')

        nodes = G_p.nodes(data=True)
        ps = np.array([i[1]['o'] for i in nodes])

        # ps = np.array(vertices)
        ax[0].plot(ps[:, 1], ps[:, 0], 'r.')
        ax[0].set_title(f'Proposal Graph {title}')
        ax[0].set_xlim(0, 1300)
        ax[0].set_ylim(0, 1300)
        ax[0].invert_yaxis()

        for (s, e) in G_t.edges():
            vals = flatten([[v] for v in G_t[s][e].values()])
            for val in vals:
                ps = val.get('pts', [])
                ax[1].plot(ps[:, 1], ps[:, 0], 'green')

        nodes = G_t.nodes(data=True)
        ps = np.array([i[1]['o'] for i in nodes])

        # ps = np.array(vertices)
        ax[1].plot(ps[:, 1], ps[:, 0], 'r.')
        #ax[1].set_title(f'Ground truth graph {title}')
        # ax[1].set_title(f'Post-Processed Graph {title}')
        # ax[1].set_xlim(0, 1300)
        # ax[1].set_ylim(0, 1300)
        # ax[1].invert_yaxis()

    if G_t is None:
        print('one graph')
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))

        for (s, e) in G_p.edges():
            vals = flatten([[v] for v in G_p[s][e].values()])
            for val in vals:
                ps = val.get('pts', [])
                ax.plot(ps[:, 1], ps[:, 0], 'green')

        nodes = G_p.nodes(data=True)
        ps = np.array([i[1]['o'] for i in nodes])

        # ps = np.array(vertices)
        ax.plot(ps[:, 1], ps[:, 0], 'r.')
        # ax.set_title(f'Proposal graph {title}')
        # ax.set_xlim(0, 1300)
        # ax.set_ylim(0, 1300)
        plt.xticks([])
        plt.yticks([])

    if save:
        plt.savefig(f'{path}/{title}.png')
    if show:
        plt.show()
    return

def plot_corresponding_graphs(path_gt_graphs, path_segmentation, show, save, path, folder_name, number_of_random_plots):
    # plot two corresponding graphs

    base = f'{path_segmentation}/graphs_not_postprocessed/'
    graphs = glob.glob(base + '*.pickle')
    value = number_of_random_plots/len(graphs)
    for graph in tqdm(graphs):
        if random.random() <= value:
            gt = graph.split('/')[-1].replace("graphs_not_postprocessed\\", '')
            if os.path.exists(path_gt_graphs + gt):
                with open(graph, "rb") as openfile:
                    G = pickle.load(openfile)

                with open(path_gt_graphs + gt, "rb") as openfile:
                    G_gt = pickle.load(openfile)
            plot_graph(G_p=G, G_t=G_gt, show=show, save=save, path=f'{path}/{folder_name}', title=os.path.splitext(gt)[0])
    return

def plot_fractured():
    # plot the fractured image of Vegas

    img = master_save_path+'fractured_vegas2.png'
    save_img = master_save_path+'fractured_vegas_plot.png'
    fig1, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(Image.open(img))
    # Set the title
    #ax.set_title("Distribution of all images of the Las Vegas dataset", fontsize=20)

    # Remove ticks on the axes
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(save_img)
    plt.show()
    return

def plot_json_metrics(source, save, show):
    # plot all metrics distribution as histograms

    for fjson in os.listdir(source):
        if '.json' in fjson:
            fig1, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.grid(which='major', linestyle='--', alpha=0.5)
            with open(source+fjson) as f:
                data = json.load(f)

            if 'GED' in fjson:
                ax.hist(list(data.values()), bins=50, density=True, color='tab:olive')
                ax.set_title("Distribution of all GED values", fontsize=20)
                ax.set_xlabel('Values')
                ax.set_ylabel('Frequency')
                plt.savefig(save+source.split('/')[-2]+'_'+fjson.replace('json', 'png'))
            if 'statistics' in fjson and 'comb' in fjson:
                vals = [i['combined'] for i in list(data.values()) if i['combined'] is not None]
                ax.hist(vals, bins=50, density=True, color='tab:olive')
                ax.set_xlim(0,1)
                ax.set_title("Distribution of all combined path length similarity values", fontsize=20)
                ax.set_xlabel('Values')
                ax.set_ylabel('Frequency')
                plt.savefig(save+source.split('/')[-2]+'_'+fjson.replace('json', 'png'))
            if show:
                plt.show()
    return

def plot_correlation():
    # plot correlation of F1 to GED and F1 to Topology

    f1 = np.array([
67.43,
61.57,
70.94,
70.57,
82.89,
87.12,
85.99,
83.66,
91.44,
91.44,
90.99,
87.31,
87.97
])
    ged = np.array([
65.75,
64.27,
72.97,
65.76,
65.58,
59.84,
60.37,
65.62,
56.81,
60.25,
57.76,62.41,
62.29
])
    topo = np.array([
71.53,
72.07,
68.03,
68.63,
79.64,
70.87,
64.2,
79.21,
87.89,
86.66,
87.1
])
    topo_new = np.array([81.71,82.09,80.47,81.01,85.19,86.47,85.82,85.17,90.07,89.08,89.00,87.47,87.67])
    print(len(f1),len(ged),len(topo_new))
    slopeg, interceptg = np.polyfit(f1, ged, 1)
    slopet, interceptt = np.polyfit(f1, topo_new, 1)
    x_approx = np.linspace(min(f1), max(f1), 100)
    gy_approx = slopeg * x_approx + interceptg
    ty_approx = slopet * x_approx + interceptt

    fig1, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].scatter(f1, ged, color='tab:orange')
    ax[0].plot(x_approx, gy_approx, 'r', label='Linear approximation', color='tab:blue')
    ax[0].set_xlabel('F1 in %', fontsize=20)
    ax[0].set_ylabel('GED in steps', fontsize=20)
    ax[1].scatter(f1, topo_new, color='tab:orange')
    ax[1].plot(x_approx, ty_approx, 'r', label='Linear approximation', color='tab:blue')
    ax[1].set_xlabel('F1 in %', fontsize=20)
    ax[1].set_ylabel('Topology in %', fontsize=20)
    plt.subplots_adjust(hspace=0.5)
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    plt.savefig(master_save_path + 'correlation.png')
    #plt.show()

def plot_F1_image(gtf, propf):
    # plot the F1 metric on an image

    gt = np.asarray(Image.open(gtf))
    prop = np.asarray(Image.open(propf))[8:1308, 8:1308]
    print(gt.shape, prop.shape, np.unique(gt), np.unique(prop))
    arr = np.full((1300, 1300), 0)

    arr[np.logical_and(prop == 255, gt == 1)] = 255
    arr[np.logical_and(prop == 255, gt == 0)] = 170
    arr[np.logical_and(prop == 0, gt == 1)] = 85

    cv2.imwrite(master_save_path+'F1_tp_fp_fn.png', arr)
    # fig1, ax = plt.subplots(1, 1, figsize=(10, 10))
    # ax.imshow(arr, cmap='gray')
    # plt.show()

# defined variables

master_save_path = 'D:/SHollendonner/graphics/'
#gt_graph_path = 'D:/SHollendonner/not_tiled/graphs/'
name = '1305_512_unet_densenet201_MS_150epochs_small' #'2605_512_unet_densenet201_MS_200epochs_full_continued' #'1305_512_unet_densenet201_MS_150epochs_small'
gt_graph_path = f'D:/SHollendonner/segmentation_results/{name}/graphs/'
seg_result_path = f'D:/SHollendonner/segmentation_results/{name}/'
all_models = ['0705_512_unet_densenet201_RGB_150epochs_small', '0805_512_unet_densenet201_RGB_150epochs_small_RESULTSFORSMALLSET', '0905_512_unet_densenet201_RGB_150epochs_small_rotatetd', '1105_512_unet_attention_RGB_150epochs_small', '1105_512_unet_attention_RGB_150epochs_small_rotation', '1305_512_unet_densenet201_MS_150epochs_small', '1405_512_unet_densenet201_MS_150epochs_small', '1605_512_unet_densenet201_MS_200epochs_full', '2104_256_basic_unet_densenet201_RGB_150epochs', '2304_256real_basic_unet_densenet201_RGB_150epochs' , 'unet3plus_1804_256_basic_unet_deepnet_RGB_200epochs', 'unet3plus_1504_256_att_unet_RGB_200epochs']

if not os.path.exists(f'{master_save_path}/{name}'):
    os.mkdir(f'{master_save_path}/{name}')

# plot F1 tp, fn, fp, tn
#plot_F1_image('D:\SHollendonner/not_tiled/rehashed/AOI_4_Shanghai_PS-RGB_img1185.png', 'img_postproc_wo_dilation.png')


# plot the json metrics distribution
"""for name in tqdm(all_models):
    seg_result_path = f'D:/SHollendonner/segmentation_results/{name}/'
    plot_json_metrics(seg_result_path, master_save_path+'performance_histograms/', show=False)"""


# plot the fractured distributiin of las vegas images
#plot_fractured()


# plot correlation between F!, GED and Topology
#plot_correlation()


# plot a graphic showing the overlap
#overlap_graphic(size=256, show=True, save=True, path=master_save_path)


# plot corresponfing graphs
#plot_corresponding_graphs(path_gt_graphs=gt_graph_path, path_segmentation=seg_result_path, number_of_random_plots=10,show=False, save=True, path=master_save_path, folder_name=name)


# plot selected graph
"""with open(seg_result_path+'graphs_not_postprocessed/AOI_4_Shanghai_PS-MS_img472.pickle', "rb") as openfile:
    G_before = pickle.load(openfile)
with open(gt_graph_path+'/AOI_4_Shanghai_PS-MS_img472.pickle', "rb") as openfile:
    G_after = pickle.load(openfile)
    
plot_graph(G_p=G_before, show=True, save=True, path=master_save_path, title=f'AOI_4_Shanghai_PS-MS_img472 before application of post-processing steps')
plot_graph(G_p=G_after, show=True, save=True, path=master_save_path, title=f'AOI_4_Shanghai_PS-MS_img472 after application of post-processing steps')"""
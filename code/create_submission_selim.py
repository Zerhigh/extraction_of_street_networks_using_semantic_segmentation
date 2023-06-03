import cv2
import numpy as np
import sknw
from pygeoif import LineString
from scipy import ndimage
from scipy.ndimage import binary_dilation
from shapely.geometry import LineString, Point
from shapely import length
from simplification.cutil import simplify_coords
from skimage.filters import gaussian
from skimage.morphology import remove_small_objects, skeletonize
import copy

"""
    script adapted from selim_sef's solutiom to spacenet challenge 3
    used for postprocessing stitched images and creating the spacenet challenge 3 submission file
    https://github.com/SpaceNetChallenge/RoadDetector/blob/0a7391f546ab20c873dc6744920deef22c21ef3e/selim_sef-solution/tools/vectorize.py
"""

def to_line_strings(mask, sigma=0.5, threashold=0.3, small_obj_size1=300, dilation=25, return_ske=False):  # , dilation=6 treshhold = 0.3
    mask = gaussian(mask, sigma=sigma)
    mask = copy.deepcopy(mask)

    mask[mask < threashold] = 0
    mask[mask >= threashold] = 1
    mask = np.array(mask, dtype="uint8")
    mask = cv2.copyMakeBorder(mask, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

    if dilation > 0:
        mask = binary_dilation(mask, iterations=dilation)
    mask, _ = ndimage.label(mask)
    mask = remove_small_objects(mask, small_obj_size1)
    mask[mask > 0] = 1
    # ret_image = mask.copy()
    # ret_image[ret_image>0] = 255
    # cv2.imwrite('img_postproc_wo_dilation.png', ret_image)

    base = np.zeros((1300, 1300))
    ske = np.array(skeletonize(mask), dtype="uint8")

    base += mask[8:1308, 8:1308]

    if return_ske:
        ske[ske > 0] = 255
        return np.uint8(ske)

    graph = sknw.build_sknw(ske, multi=True)

    line_strings = []
    lines = []
    all_coords = []
    nodes = graph.nodes()
    # draw edges by pts
    for (s, e) in graph.edges():
        for k in range(len(graph[s][e])):
            ps = graph[s][e][k]['pts']
            coords = []
            start = (int(nodes[s]['o'][1]), int(nodes[s]['o'][0]))
            all_points = set()

            for i in range(1, len(ps)):
                pt1 = (int(ps[i - 1][1]), int(ps[i - 1][0]))
                pt2 = (int(ps[i][1]), int(ps[i][0]))
                if pt1 not in all_points and pt2 not in all_points:
                    coords.append(pt1)
                    all_points.add(pt1)
                    coords.append(pt2)
                    all_points.add(pt2)
            end = (int(nodes[e]['o'][1]), int(nodes[e]['o'][0]))

            same_order = True
            if len(coords) > 1:
                same_order = np.math.hypot(start[0] - coords[0][0], start[1] - coords[0][1]) <= np.math.hypot(
                    end[0] - coords[0][0], end[1] - coords[0][1])
            if same_order:
                coords.insert(0, start)
                coords.append(end)
            else:
                coords.insert(0, end)
                coords.append(start)
            coords = simplify_coords(coords, 2.0)
            # print(coords)
            all_coords.append(coords)

    # print(all_coords)
    for coords in all_coords:
        if len(coords) > 0:
            line_obj = LineString(coords)
            lines.append(line_obj)
            line_string_wkt = line_obj.wkt
            line_strings.append(line_string_wkt)
    new_lines = remove_duplicates(lines)
    new_lines = filter_lines(new_lines, calculate_node_count(new_lines))
    line_strings = [l.wkt for l in new_lines]
    lengths = [length(l) for l in new_lines]

    # return skeleton too
    ske[ske > 0] = 255

    return line_strings, lengths, np.uint8(ske), graph #, buffered_arr #ret_mask[8:1308, 8:1308]


def remove_duplicates(lines):
    all_paths = set()
    new_lines = []
    for l, line in enumerate(lines):
        points = line.coords
        for i in range(1, len(points)):
            pt1 = (int(points[i - 1][0]), int(points[i - 1][1]))
            pt2 = (int(points[i][0]), int(points[i][1]))
            if (pt1, pt2) not in all_paths and (pt2, pt1) not in all_paths and not pt1 == pt2:
                new_lines.append(LineString((pt1, pt2)))
                all_paths.add((pt1, pt2))
                all_paths.add((pt2, pt1))
    return new_lines


def filter_lines(new_lines, node_count):
    filtered_lines = []
    for line in new_lines:
        points = line.coords
        pt1 = (int(points[0][0]), int(points[0][1]))
        pt2 = (int(points[1][0]), int(points[1][1]))

        length = np.math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])

        if not ((node_count[pt1] == 1 and node_count[pt2] > 2 or node_count[pt2] == 1 and node_count[
            pt1] > 2) and length < 10):
            filtered_lines.append(line)
    return filtered_lines


def calculate_node_count(new_lines):
    node_count = {}
    for l, line in enumerate(new_lines):
        points = line.coords
        for i in range(1, len(points)):
            pt1 = (int(points[i - 1][0]), int(points[i - 1][1]))
            pt2 = (int(points[i][0]), int(points[i][1]))
            pt1c = node_count.get(pt1, 0)
            pt1c += 1
            node_count[pt1] = pt1c
            pt2c = node_count.get(pt2, 0)
            pt2c += 1
            node_count[pt2] = pt2c
    return node_count


def split_line(line):
    all_lines = []
    points = line.coords
    pt1 = (int(points[0][0]), int(points[0][1]))
    pt2 = (int(points[1][0]), int(points[1][1]))
    dist = np.math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])
    if dist > 10:
        new_lines = cut(line, 5)
        for l in new_lines:
            for sl in split_line(l):
                all_lines.append(sl)
    else:
        all_lines.append(line)
    return all_lines


def cut(line, distance):
    # Cuts a line in two at a distance from its starting point
    # This is taken from shapely manual
    if distance <= 0.0 or distance >= line.length:
        return [LineString(line)]
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pd = line.project(Point(p))
        if pd == distance:
            return [
                LineString(coords[:i + 1]),
                LineString(coords[i:])]
        if pd > distance:
            cp = line.interpolate(distance)
            return [
                LineString(coords[:i] + [(cp.x, cp.y)]),
                LineString([(cp.x, cp.y)] + coords[i:])]

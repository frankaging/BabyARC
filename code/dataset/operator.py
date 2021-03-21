import os
import torch
from collections import Counter
from operator import itemgetter
import random
from random import sample
import numpy as np
import json
from collections import OrderedDict
import pickle
import matplotlib.pylab as plt
from functools import partial
import pprint
import hashlib
import copy
import sys
import math
from copy import deepcopy
import hashlib
import uuid 
import ast
# Baby-ARC related imports
from constants import *
from utils import *
from objects import *
from canvas import *

class OperatorEngine(object):
    
    def __init__(self):
        pass
    
    def get_relation_nbrs(self, node, edges):
        for k, v in edges.items():
            if k[0] == node:
                if v == "Attr":
                    yield (k[1], v)
                else:
                    yield (v,)

    def dfs_path_helper(self, curr_node, curr_path, result, edges, k=3):
        if len(curr_path) != 0:
            result.add(tuple(copy.deepcopy(curr_path)))
        if len(curr_path) == 3:
            return
        
        for nbr in self.get_relation_nbrs(curr_node, edges):
            if nbr not in curr_path:
                curr_path.append(nbr)
                self.dfs_path_helper(nbr, curr_path, result, edges, k=k)
                curr_path.pop()
        
    def dfs_path(self, root, edges, k=3):
        results = set([])
        self.dfs_path_helper(root, [], results, edges, k=k)
        return results
    
    def get_obj_dfs_path(self, canvas, k=3):
        # extrapolate the relation to be bidirectional for all relations
        relation_edges = canvas.partial_relation_edges
        relation_edges_copy = copy.deepcopy(relation_edges)
        for k, v in relation_edges.items():
            if v != "Attr":
                relation_edges_copy[(k[1], k[0])] = v
        parsed_relation_edges = canvas.parse_relations()
        for k, v in parsed_relation_edges.items():
            if v == "Attr":
                relation_edges_copy[(canvas.id_node_map[k[0]], k[1])] = v
            else:
                relation_edges_copy[(canvas.id_node_map[k[0]], canvas.id_node_map[k[1]])] = v
        
        # loop through each obj
        obj_paths = OrderedDict({})
        common_in = None
        for k, v in canvas.id_node_map.items():
            v_paths = self.dfs_path(v, relation_edges_copy, k=k)
            if common_in == None:
                common_in = v_paths
            else:
                common_in = common_in.intersection(v_paths)
            obj_paths[tuple([v])] = v_paths
        for k, v in obj_paths.items():
            obj_paths[k] = v - common_in
        return obj_paths
    
    def select_by_common_referred_patterns(self, canvas_list):
        """
        extract patterns exist 
        """
        common_referred = None
        for canvas in canvas_list:
            if common_referred == None:
                common_referred = self.get_obj_dfs_path(canvas)
            else:
                new_common_referred = OrderedDict({})
                current_referred = self.get_obj_dfs_path(canvas)
                for k_common, v_common in common_referred.items():
                    for k_current, v_current in current_referred.items():
                        v_shared = v_common.intersection(v_current)
                        if len(v_shared) > 0:
                            new_k = tuple(list(k_common) + list(k_current))
                            new_common_referred[new_k] = v_shared
                common_referred = copy.deepcopy(new_common_referred)
        selectors = []
        for k, v in common_referred.items():
            selectors.append([[k[0]], [k[1]]])
        return selectors

    """
    object-return selector
    """
    def select_by_shape_prior(self, canvas_list, shape="pixel"):
        # extrapolate the relation to be bidirectional for all relations
        relation_edges = canvas.partial_relation_edges
        relation_edges_copy = copy.deepcopy(relation_edges)
        for k, v in relation_edges.items():
            if v != "Attr":
                relation_edges_copy[(k[1], k[0])] = v
        parsed_relation_edges = canvas.parse_relations()
        for k, v in parsed_relation_edges.items():
            if v == "Attr":
                relation_edges_copy[(canvas.id_node_map[k[0]], k[1])] = v
            else:
                relation_edges_copy[(canvas.id_node_map[k[0]], canvas.id_node_map[k[1]])] = v
    
    def select_by_color_prior(self, canvas_list, color=[1]):
        pass
    
    def select_by_boundary_prior(self, canvas_list, boundary=[1]):
        pass
    
    """
    numeric-return selector
    """
    def select_by_extreme_size(self, extreme="smallest"):
        pass
    
    def select_by_extreme_position(self, extreme="topleft"):
        pass
    
    def select_by_pixel_count(self, count_by="color"):
        pass
    
    def get_exclude_colors(self, canvas_list, objs_list=[]):
        pass
    
    """
    basic operators
    "Identity", "hFlip", "vFlip", "RotateA", "RotateB", "RotateC", "DiagFlipA", "DiagFlipB"
    """
    def operator_by_basic(self, canvas_list, objs_list=[], operator="hFlip"):
        pass
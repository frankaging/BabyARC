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
try:
    from .constants import *
    from .utils import *
    from .objects import *
    from .canvas import *
except:
    from constants import *
    from utils import *
    from objects import *
    from canvas import *

class OperatorEngine(object):
    
    def __init__(self):
        
        # DSLs for operators
        self.ARG_TYPES = {"$OBJ", "$NUMERIC", "$WHOLE", "$POSITION", "$COLOR"} # <- produced by selector
        self.OPERATORS = {
            "@IDENTITY" : "$OBJ|$WHOLE",
            "@ROTATION" : "$OBJ|$WHOLE", # this rotation contains flip as well!
            # "@DRAWLINE" : "$POSITION&$POSITION&$NUMERIC",
            "@DRAW" : "$OBJ&$COLOR|$POSITION&$COLOR|$NUMERIC&$COLOR",
            # "@DRAWPATTERN" : "$POSITION&$NUMERIC",
            # "@MOVE" : "$OBJ&$POSITION",
            # "@REMAINDER" : "$OBJ",
            # "@SORT" : "$OBJ&$NUMERIC",
            "@OUT" : "$OBJ|$WHOLE",
            "@IN" : "$WHOLE"
        }
        self.SELECTORS = {
            "$OBJ" : ["$SELECTOR_PATTERN", "$SELECTOR_ALL", "$SELECTOR_EXTREME"], 
            "$WHOLE" : ["$SELECTOR_WHOLE"],
            "$COLOR" : ["$OBJ^$SELECTOR_COLOR"],
            "$NUMERIC" : ["$OBJ^$SELECTOR_NUMERIC"],
            "$POSITION" : ["$OBJ^$SELECTOR_POSITION"]
        }
        
        for selector, patterns in self.SELECTORS.items():
            new_patterns = []
            for p in patterns:
                if "^" in p:
                    
                    # high order selector, assumes it goes only second order!
                    p = p.split("^")
                    for ele in self.SELECTORS[p[0]]:
                        new_patterns.append(f"{ele}^{p[-1]}")
                else:
                    new_patterns.append(p)
            self.SELECTORS[selector] = new_patterns
    
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
    
    def _consolidate_patterns_within_canvas(self, obj_paths):
        reversed_patterns = OrderedDict({})
        object_pool = []
        for k, _ in obj_paths.items():
            object_pool.append(k)

        for i in range(len(object_pool)):
            obj_i = object_pool[i]
            obj_i_path = obj_paths[obj_i]
            for path in obj_i_path:
                matched = False
                for k, v in reversed_patterns.items():
                    # consolidate in
                    if k == path:
                        reversed_patterns[path].add(obj_i[0]) # this is only for a single obj
                        matched = True
                        break
                # adding this to the reversed pattern mapping as well
                if not matched:
                    reversed_patterns[path] = set([obj_i[0]])
        return reversed_patterns
    
    def get_obj_dfs_path(self, canvas, k=3):
        # extrapolate the relation to be bidirectional for all relations
        relation_edges = canvas.partial_relation_edges
        relation_edges_copy = copy.deepcopy(relation_edges)
        for k, v in relation_edges.items():
            if v != "Attr" and v != "IsInside": # IsInside is not permutable here!
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
        return self._consolidate_patterns_within_canvas(obj_paths) # return consolidated paths

    def _sample_operator_combination(self, k=2, middle=True):
        avaliable_operators = set(list(self.OPERATORS.keys()))
        if middle:
            avaliable_operators = avaliable_operators - {"@OUT", "@IN"}
        if k > 1:
            avaliable_operators = avaliable_operators - {"@COMBINED", "@REMAINDER", "@SORT"}
        existing_operators = list(avaliable_operators)
        operator_permutator = []
        for i in range(k):
            operator_permutator.append(copy.deepcopy(existing_operators))
        operator_permutations = product(*operator_permutator)
        operator_combinations = []
        for perm in operator_permutations:
            if len(set(perm)) == len(perm): # non-repeat
                operator_combinations.append(perm)
        return operator_combinations 
    
    def _graph_convertor(self, flatten_graph):
        graph = OrderedDict({})
        for k, v in flatten_graph.items():
            if v in graph.keys():
                graph[v].append(k)
            else:
                graph[v] = [k]
        return graph

    def sample_operator_graph(self, max_ops_depth=3):
        operator_graphs = []
        avaliable_operators = set(list(self.OPERATORS.keys()))
        avaliable_operators = avaliable_operators - {"@OUT", "@IN"}
        op_graph_pemutator = []
        for i in range(max_ops_depth):
            op_graph_pemutator.append(copy.deepcopy(list(avaliable_operators)))
        op_graphs = product(*op_graph_pemutator)
        for op_g in op_graphs:
            args_permutator = []
            for steps in op_g:
                step_args = self.OPERATORS[steps].split("|") # there could be potentially multiple args
                                                             # supported for this ops
                args_permutator.append(step_args)
            args_permutations = product(*args_permutator)
            for step_args in args_permutations:
                step_idx = 1
                curr_step_permutator = []
                for s_a in step_args:
                    step_perms = []
                    s_a = s_a.split("&")
                    
                    # get all the s_a combinations
                    s_a_permutators = []
                    for s_a_s in s_a:
                        s_a_permutators.append(self.SELECTORS[s_a_s])
                    s_a_permutations = product(*s_a_permutators)
                    for s_a_perm in s_a_permutations:
                        # potential output connecting points
                        out_in = []
                        for i in range(step_idx):
                            out_in.append(f"{i}$OUT")
                        out_permutator = []
                        for j in range(len(s_a_perm)):
                            out_permutator.append(out_in)
                        out_permutations = product(*out_permutator)
                        for out_perm in out_permutations:
                            v = op_g[step_idx-1]
                            for combo in zip(out_perm, s_a_perm):
                                v += f" {combo[0]}^{combo[1]}"
                            if f"{step_idx-1}$OUT" in v:
                                step_perms.append(v) # you need to include last one!
                    curr_step_permutator.append(step_perms)
                    step_idx += 1
                curr_step_permutations = product(*curr_step_permutator)
                for c_s_p in curr_step_permutations:
                    graph = OrderedDict({})
                    graph["0$OUT"] = "@IDENTITY 0$IN"
                    for i in range(len(c_s_p)):
                        graph[f"{i+1}$OUT"] = c_s_p[i]
                    if graph[f"{len(c_s_p)}$OUT"].split(" ")[0] == "@IDENTITY" and \
                        "^".join(graph[f"{len(c_s_p)}$OUT"].split("^")[1:]) in self.SELECTORS["$NUMERIC"]:
                        graph[f"{len(c_s_p)+1}$OUT"] = f"@DRAW {len(c_s_p)}$OUT"
                        operator_graphs.append(copy.deepcopy(graph))
                    else:
                        for o_perm in self.SELECTORS["$OBJ"]+self.SELECTORS["$WHOLE"]:
                            graph[f"{len(c_s_p)+1}$OUT"] = f"@IDENTITY {len(c_s_p)}$OUT^{o_perm}"
                            operator_graphs.append(copy.deepcopy(graph))
        return operator_graphs
    
    def _count_image_t_pixel(self, image_t, background_color=0):
        count = 0
        for i in range(image_t.shape[0]):
            for j in range(image_t.shape[1]):
                if image_t[i,j] != background_color:
                    count += 1
        return count
    
    def _generate_ranking_tuples(self, canvas, rank_by="#POS_COL"): # output single object
        ranking_tuples = []
        for k, v in canvas.node_id_map.items():
            position_t = canvas.opos_map[v]
            row = position_t[0]
            col = position_t[1]
            obj = canvas.oid_map[v]
            h = obj.image_t.shape[0]
            w = obj.image_t.shape[1]
            c_h = canvas.init_canvas.shape[0]
            c_w = canvas.init_canvas.shape[1]
            if rank_by == "#POS_COL":
                ranking_tuples.append((k, col))
            elif rank_by == "#POS_COL_R":
                ranking_tuples.append((k, c_w-(col+w)))
            elif rank_by == "#POS_ROW":
                ranking_tuples.append((k, row))
            elif rank_by == "#POS_ROW_R":
                ranking_tuples.append((k, c_h-(row+h)))
            elif rank_by == "#AREA":
                ranking_tuples.append((k, h*w))
            elif rank_by == "#WIDTH":
                ranking_tuples.append((k, w))
            elif rank_by == "#HEIGHT":
                ranking_tuples.append((k, h))
            elif rank_by == "#COUNT_PIXEL":
                ranking_tuples.append((k, self._count_image_t_pixel(obj.image_t, canvas.background_color)))
        
        return sorted(ranking_tuples, key=itemgetter(1))

    def _rotate_tag(self, original_tags, n):
        """
        rotate the tag ccw
        """
        curr_position_tags = copy.deepcopy(original_tags)
        for i in range(n):
            new_position_tags = []
            for t in curr_position_tags:
                if t == "upper":
                    new_position_tags.append("left")
                elif t == "left":
                    new_position_tags.append("lower")
                elif t == "lower":
                    new_position_tags.append("right")
                elif t == "right":
                    new_position_tags.append("upper")
                else:
                    pass
            curr_position_tags = copy.deepcopy(new_position_tags) # recurrent
        return curr_position_tags
    
    def _flip_tag(self, original_tags, direction):
        curr_position_tags = copy.deepcopy(original_tags)
        if direction == -1:
            new_position_tags = []
            for t in curr_position_tags:
                if t == "left":
                    new_position_tags.append("right")
                elif t == "right":
                    new_position_tags.append("left")
                else:
                    new_position_tags.append(t)
        elif direction == -2:
            new_position_tags = []
            for t in curr_position_tags:
                if t == "upper":
                    new_position_tags.append("lower")
                elif t == "lower":
                    new_position_tags.append("upper")
                else:
                    new_position_tags.append(t)
        return new_position_tags

    def _rotation_obj(self, img_obj, rotation):
        ret = copy.deepcopy(img_obj)
        # update image and tag
        if rotation == 0:
            ret.image_t = ret.image_t.flip(-1)
            ret.position_tags = self._flip_tag(ret.position_tags, -1)
        elif rotation == 1:
            ret.image_t = ret.image_t.flip(-2)
            ret.position_tags = self._flip_tag(ret.position_tags, -2)
        elif rotation == 2:
            ret.image_t = torch.rot90(ret.image_t, k=1, dims=(-2, -1))
            ret.position_tags = self._rotate_tag(ret.position_tags, 1)
        elif rotation == 3:
            ret.image_t = torch.rot90(ret.image_t, k=2, dims=(-2, -1))
            ret.position_tags = self._rotate_tag(ret.position_tags, 2)
        elif rotation == 4:
            ret.image_t = torch.rot90(ret.image_t, k=3, dims=(-2, -1))
            ret.position_tags = self._rotate_tag(ret.position_tags, 3)
        elif rotation == 5:
            ret.image_t = torch.rot90(ret.image_t, k=1, dims=(-2, -1)).flip(-1)
            ret.position_tags = self._flip_tag(self._rotate_tag(ret.position_tags, 1), -1)
        elif rotation == 6:
            ret.image_t = torch.rot90(ret.image_t, k=1, dims=(-2, -1)).flip(-2)
            ret.position_tags = self._flip_tag(self._rotate_tag(ret.position_tags, 1), -2)
        else:
            pass
        return ret
    
    def _rotate_img_t(self, img_t_in, rotation):
        img_t = copy.deepcopy(img_t_in)
        if rotation == 0:
            img_t = img_t.flip(-1)
        elif rotation == 1:
            img_t = img_t.flip(-2)
        elif rotation == 2:
            img_t = torch.rot90(img_t, k=1, dims=(-2, -1))
        elif rotation == 3:
            img_t = torch.rot90(img_t, k=2, dims=(-2, -1))
        elif rotation == 4:
            img_t = torch.rot90(img_t, k=3, dims=(-2, -1))
        elif rotation == 5:
            img_t = torch.rot90(img_t, k=1, dims=(-2, -1)).flip(-1)
        elif rotation == 6:
            img_t = torch.rot90(img_t, k=1, dims=(-2, -1)).flip(-2)
        else:
            pass
        return img_t
    
    def _rotate_whole_obj_position(self, canvas_t_in, obj_t_in, old_pos_in, rotation, background_color):
        canvas_t = copy.deepcopy(canvas_t_in)
        obj_t = copy.deepcopy(obj_t_in)
        old_pos = copy.deepcopy(old_pos_in)
        for i in range(obj_t.shape[0]):
            for j in range(obj_t.shape[1]):
                r = old_pos[0] + i
                c = old_pos[1] + j
                canvas_t[r,c] = 1.0
        canvas_t = self._rotate_img_t(canvas_t, rotation)
        for i in range(canvas_t.shape[0]):
            for j in range(canvas_t.shape[1]):
                if canvas_t[i,j] != background_color:
                    return torch.tensor([i,j])
    
    def _rotate_whole(self, canvas, rotation):
        new_canvas = copy.deepcopy(canvas)
        new_init_canvas = self._rotate_img_t(new_canvas.init_canvas, rotation)
        new_canvas.init_canvas = new_init_canvas
        for oid, obj in new_canvas.oid_map.items():
            rotated_obj = self._rotation_obj(obj, rotation)
            new_canvas.oid_map[oid] = rotated_obj
            # then we need to find the new position on the new canvas!
            new_pos = \
                self._rotate_whole_obj_position(canvas.init_canvas, obj.image_t, canvas.opos_map[oid],
                                                rotation,
                                                canvas.background_color)
            new_canvas.opos_map[oid] = new_pos
        return new_canvas
    
    def _rotate_obj_position(self, obj_t_in, old_pos_in, rotation, center_rotate=False):
        """
        Rotate by the center of the object.
        """
        center_r = old_pos_in[0] + int(obj_t_in.shape[0]/2)
        center_c = old_pos_in[1] + int(obj_t_in.shape[1]/2)
        
        new_obj_t = self._rotate_img_t(obj_t_in, rotation)
        new_r = center_r - int(new_obj_t.shape[0]/2)
        new_c = center_c - int(new_obj_t.shape[1]/2)
        
        if center_rotate:
            return torch.tensor([new_r, new_c]), new_obj_t
        else:
            return torch.tensor([old_pos_in[0], old_pos_in[1]]), new_obj_t
    
    def _select_position_obj(self, canvas, obj, condition):
        _id = canvas.node_id_map[obj]
        position = canvas.opos_map[_id]
        p_r = position[0]
        p_c = position[1]
        obj = canvas.oid_map[_id]
        r = obj.image_t.shape[0]
        c = obj.image_t.shape[1]
        if condition=="#UPPER_LEFT":
            return torch.tensor([p_r, p_c])
        elif condition=="#UPPER_RIGHT":
            return torch.tensor([p_r, p_c+c])
        elif condition=="#LOWER_LEFT":
            return torch.tensor([p_r+r, p_c])
        elif condition=="#LOWER_RIGHT":
            return torch.tensor([p_r+r, p_c+c])
    
    def _select_numeric_canvas(self, canvas, obj_selector, condition):
        if condition == "#OBJ_COUNT":
            return [len(obj_selector)]
    
    ########################################
    #
    # Selectors
    #
    ########################################
    def select_by_common_referred_patterns(self, canvas_list):
        """
        extract patterns exist 
        """
        common_referred = None
        for canvas in canvas_list:
            if common_referred == None:
                common_referred = self.get_obj_dfs_path(canvas)
                # turning value into 2d list to contain multiple canvas information
                flatten_common_referred = OrderedDict({})
                for k, v in common_referred.items():
                    flatten_common_referred[k] = [v]
                common_referred = flatten_common_referred
            else:
                new_common_referred = OrderedDict({})
                current_referred = self.get_obj_dfs_path(canvas) # we dont have to flatten it!
                for path_common, objs_common in common_referred.items():
                    for path_current, objs_current in current_referred.items():
                        if path_common == path_current:
                            new_common_referred[path_common] = []
                            new_common_referred[path_common].extend(objs_common)
                            # common path shared across two canvas
                            new_common_referred[path_common].append(objs_current)
                common_referred = copy.deepcopy(new_common_referred)
        selectors = []
        # return potential selectors as a 2d list, same length as the canvas
        for _, v in common_referred.items():
            # we ignore the keys here for significant reasons.
            # we don't want to explicity expose the relations to 
            # users.
            ss = []
            for e in v:
                ss += [list(e)]
            selectors += [ss]
        return selectors

    def select_by_extreme(self, canvas_list):
        extreme_code = {
            0:"#POS_COL",
            1:"#POS_COL_R",
            2:"#POS_ROW",
            3:"#POS_ROW_R",
            4:"#AREA",
            5:"#WIDTH",
            6:"#HEIGHT",
            7:"#COUNT_PIXEL"
        }
        selector_code = random.randint(0, len(list(extreme_code.keys()))-1)
        selectors = []
        min_len = 99
        for canvas in canvas_list:
            selected_objs = self._generate_ranking_tuples(canvas, rank_by=extreme_code[selector_code])
            if len(selected_objs) < min_len:
                min_len = len(selected_objs)
            selectors.append(selected_objs)
        selector_idx = random.randint(0, min_len-1)
        single_selectors = []
        for s in selectors:
            single_selectors.append([s[selector_idx][0]])
        return single_selectors, extreme_code[selector_code]
    
    def select_all(self, canvas_list):
        selectors = []
        for canvas in canvas_list:
            canvas_selector = []
            for k, v in canvas.node_id_map.items():
                canvas_selector.append(k)
            selectors.append(canvas_selector)
        return selectors
    
    def select_position(self, canvas_list, obj_selectors=[]):
        """
        For each canvas, we should just return a sinple position
        """
        position_code = {
            0:"#UPPER_LEFT",
            1:"#UPPER_RIGHT",
            2:"#LOWER_LEFT",
            3:"#LOWER_RIGHT",
        }
        selectors = []
        selector_code = random.randint(0, len(list(position_code.keys()))-1)
        for i in range(len(canvas_list)):
            selected_position = self._select_position_obj(canvas_list[i], obj_selectors[i][0], 
                                                          position_code[selector_code])
            selectors.append([selected_position])
        return selectors
    
    def select_numeric(self, canvas_list, obj_selectors=[]):      
        numeric_code = {
            0:"#OBJ_COUNT",
            1:"#RANDOM_NUMERIC"
        }
        selectors = []
        selector_code = random.randint(0, len(list(numeric_code.keys()))-1)
        if numeric_code[selector_code] == "#RANDOM_NUMERIC":
            randint = random.randint(1,9)
            for i in range(len(canvas_list)):
                selectors.append([randint])
        else:
            for i in range(len(canvas_list)):
                selector = self._select_numeric_canvas(canvas_list[i], obj_selectors[i], 
                                                       numeric_code[selector_code])
                selectors.append(selector)
        return selectors
    
    def _select_color_canvas(self, canvas, obj_selector, condition):
        color_freq_map = {}
        for obj in obj_selector:
            _id = canvas.node_id_map[obj]
            image_t = canvas.oid_map[_id].image_t
            for i in range(image_t.shape[0]):
                for j in range(image_t.shape[1]):
                    if image_t[i,j] != canvas.background_color:
                        if image_t[i,j] not in color_freq_map.keys():
                            color_freq_map[image_t[i,j]] = 1
                        else:
                            color_freq_map[image_t[i,j]] += 1
        if condition == "#MOST_FREQ_COLOR":
            pass        
        elif condition == "#LEAST_FREQ_COLOR":
            pass

    def select_color(self, canvas_list, obj_selectors=[]):
        """
        select color can be random, or based on the object selected.
        """
        color_code = {
            0:"#MOST_FREQ_COLOR",
            1:"#LEAST_FREQ_COLOR",
            1:"#RANDOM_COLOR"
        }
        selectors = []
        selector_code = random.randint(0, len(list(color_code.keys()))-1)
        if color_code[selector_code] == "#RANDOM_COLOR":
            randint = random.randint(1,9)
            for i in range(len(canvas_list)):
                selectors.append([randint])
        else:
            for i in range(len(canvas_list)):
                selector = self._select_color_canvas(canvas_list[i], obj_selectors[i], 
                                                     color_code[selector_code])
                selectors.append(selector)
        return selectors

    ########################################
    #
    # Operators
    #
    ########################################
    def operator_rotate_whole(self, canvas_list, selectors=[], selector_type="$WHOLE"):
        """
        Rotate whole canvas, selector is not needed!
        """
        rotate_code = {
            0:"#hFlip",
            1:"#vFlip",
            2:"#RotateA",
            3:"#RotateB",
            4:"#RotateC",
            5:"#DiagFlipA",
            6:"#DiagFlipB",
        }
        selector_code = random.randint(0, len(list(rotate_code.keys()))-1)
        operated_canvas = []
        for canvas in canvas_list:
            new_canvas = self._rotate_whole(canvas, selector_code)
            operated_canvas.append(new_canvas)
        return operated_canvas, rotate_code[selector_code]
    
    def operate_rotate(self, canvas_list, selectors=[],
                       operator_tag=None, allow_connect=True, 
                       allow_shape_break=False, selector_type="$OBJ"):
        rotate_code = {
            0:"#hFlip",
            1:"#vFlip",
            2:"#RotateA",
            3:"#RotateB",
            4:"#RotateC",
            5:"#DiagFlipA",
            6:"#DiagFlipB",
        }
        rotate_code_reverse = {}
        for k,v in rotate_code.items():
            rotate_code_reverse[v] = k
        if operator_tag:
            selector_code = rotate_code_reverse[operator_tag]
            operated_canvas = []
            canvas_idx = 0
            for canvas in canvas_list:
                new_canvas = copy.deepcopy(canvas)
                for selected_obj in selectors[canvas_idx]:
                    _id = canvas.node_id_map[selected_obj]
                    old_obj = canvas.oid_map[_id]
                    old_pos = canvas.opos_map[_id]
                    new_pos, new_obj_img_t = self._rotate_obj_position(old_obj.image_t, old_pos, selector_code)
                    if not allow_shape_break:
                        if new_pos[0] < 0 or new_pos[1] < 0:
                            return -1, operator_tag
                        if new_pos[0] + new_obj_img_t.shape[0] > new_canvas.init_canvas.shape[0] or \
                            new_pos[1] + new_obj_img_t.shape[1] > new_canvas.init_canvas.shape[1]:
                            return -1, operator_tag
                    new_canvas.oid_map[_id].image_t = new_obj_img_t
                    new_canvas.opos_map[_id] = new_pos
                    # check the canvas compliant
                    if new_canvas.check_conflict(connect_allow=allow_connect): # this is little hard, but ok?
                        operated_canvas.append(new_canvas)
                    else:
                        break
                canvas_idx += 1
            if len(operated_canvas) == len(canvas_list):
                return operated_canvas, operator_tag
            else:
                return -1, operator_tag
        else:
            retry = 5
            for i in range(retry):
                selector_code = random.randint(0, len(list(rotate_code.keys()))-1)
                operated_canvas = []
                canvas_idx = 0

                for canvas in canvas_list:
                    new_canvas = copy.deepcopy(canvas)
                    for selected_obj in selectors[canvas_idx]:
                        _id = canvas.node_id_map[selected_obj]
                        old_obj = canvas.oid_map[_id]
                        old_pos = canvas.opos_map[_id]
                        new_pos, new_obj_img_t = self._rotate_obj_position(old_obj.image_t, old_pos, selector_code)
                        
                        new_canvas.oid_map[_id].image_t = new_obj_img_t
                        new_canvas.opos_map[_id] = new_pos
                        # check the canvas compliant
                        if new_canvas.check_conflict(connect_allow=allow_connect): # this is little hard, but ok?
                            operated_canvas.append(new_canvas)
                        else:
                            break
                    canvas_idx += 1
                if len(operated_canvas) == len(canvas_list):
                    break
            if len(operated_canvas) == len(canvas_list):
                return operated_canvas, rotate_code[selector_code]
            else:
                -1, rotate_code[selector_code] # Fail to operate!
            
    def operate_draw(self, canvas_list, obj_selectors=None, color_selectors=None, position_selectors=None, 
                     numeric_selectors=None, selector_type="$OBJ&$COLOR"):
        pass
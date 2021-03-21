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

class BabyARCDataset(object):
    """
    The main class of the BabyARC Dataset。
    TODO:
    1. Support different background color.
    """
    def __init__(self, pretrained_obj_cache, 
                 data_dir=None, save_directory="./BabyARCDataset/", 
                 object_limit=None,
                 # below are basic statistics of the dataset
                 dataset_background_color=0.0,
                 relation_vocab = ["SameAll", "SameShape", "SameColor", 
                                   "SameRow", "SameCol", 
                                   "IsInside", "IsTouch"],
                 num_pool = [2,3,4],
                 sparse_prob = 0.3,
                 noise_level = 1):
        if data_dir == None:
            logger.info("Creating new BabyARC dataset by loading in pretrained objects.")
            self.training_objs = torch.load(pretrained_obj_cache)
            logger.info(f"Loading the object engine and canvas engine with "
                        "a limit of object number {object_limit}, "
                        "background_color={int(dataset_background_color)}.")
            if object_limit:
                self.ObE = ObjectEngine(self.training_objs[:object_limit], 
                                        background_c=dataset_background_color)
            else:
                self.ObE = ObjectEngine(self.training_objs, 
                                        background_c=dataset_background_color)
            
            self.CanvasE = CanvasEngine(background_color=dataset_background_color)
            self.relation_vocab = relation_vocab
            self.num_pool = num_pool
            self.sparse_prob = sparse_prob
            self.noise_level = noise_level
        self.data_dir = data_dir # if it is None, we need to generate new dataset
        self.save_directory = save_directory
    
    def sample_single_core_edges(self):
        relation_num = random.choice(self.num_pool)
        edges = OrderedDict({ })
        relation_sampled = []
        for i in range(relation_num):
            relation_sampled += [random.choice(self.relation_vocab)]
            sorted(relation_sampled)
        current_obj_count = 0
        for rel in relation_sampled:
            if rel == "IsInside":
                # we need to propose two new objects
                edges[(f'obj_{current_obj_count}', f'obj_{current_obj_count+1}')] = rel
                current_obj_count += 2
            else:
                # if others we can decide whether to connect
                if current_obj_count == 0 or random.uniform(0, 1) < self.sparse_prob:
                    # sparse
                    edges[(f'obj_{current_obj_count}', f'obj_{current_obj_count+1}')] = rel
                    current_obj_count += 2
                else:
                    # select a old obj
                    ref_obj = random.randint(0,current_obj_count-1)
                    edges[(f'obj_{current_obj_count}', f'obj_{ref_obj}')] = rel
                    current_obj_count += 1
        return edges
    
    def sample_single_canvas_by_core_edges(self, edges, retry=5, is_plot=True, min_length=20, max_length=30):
        relation_num = len(edges)
        nodes = OrderedDict({ })

        for i in range(retry):
            if relation_num >= 3 or i > (retry*0.5):
                test_canvas = CanvasEngine().sameple_canvas_by_size(min_length=min_length, max_length=max_length)[0]
            else:
                test_canvas = CanvasEngine().sameple_canvas()[0]

            placed_objs = set([])
            current_id = 0
            for edge, rel in edges.items():
                node_left = edge[0]
                node_right = edge[1]
                rel_n = rel

                # ok, we now support you specify some node attribute
                # ('obj_0', 'color_[1]', 'Attr')
                # ('obj_0', 'pos_[0,1]', 'Attr')
                if rel_n.startswith("Attr"):
                    if node_left not in placed_objs:
                        if node_right.startswith("color"):
                            new_c = node_right.split("_")[-1]
                            new_c = ast.literal_eval(new_c)
                            new_c = int(new_c[0]) # get the color
                            obj_refer = self.ObE.sample_objs(n=1, is_plot=False)[0]
                            obj_refer = self.ObE.fix_color(obj_refer, new_color=new_c)
                            placement_result = test_canvas.placement(obj_refer, consider_tag=False) # place old obj with free pos
                            if placement_result == -1:
                                break
                        elif node_right.startswith("pos"):
                            new_p = node_right.split("_")[-1]
                            new_p = ast.literal_eval(new_p)
                            if len(new_p) != 2:
                                placement_result = -1
                                break
                            obj_refer = self.ObE.sample_objs(n=1, is_plot=False)[0]
                            placement_result = test_canvas.placement_position_fixed(obj_refer, new_p[0], new_p[1])
                            if placement_result == -1:
                                break
                        elif node_right.startswith("pixel"):
                            obj_refer = self.ObE.sample_objs_with_pixel(n=1)[0]
                            placement_result = test_canvas.placement(obj_refer, consider_tag=False) # place old obj with free pos
                            if placement_result == -1:
                                break
                        elif node_right.startswith("line"):
                            line_spec = node_right.split("_")[-1]
                            line_spec = ast.literal_eval(line_spec)
                            if line_spec[0] != -1:
                                len_lims=[line_spec[0], line_spec[0]]
                            else:
                                len_lims=[2, min(test_canvas.init_canvas.shape[0], test_canvas.init_canvas.shape[1])]
                            if line_spec[1] != -1:
                                thickness = line_spec[1]
                            else:
                                thickness = 1
                            if line_spec[2] == -1:
                                directions=["v", "h"]
                                direction = random.sample(directions, k=1)[0]
                            elif line_spec[2] == 0:
                                direction = "v"
                            elif line_spec[2] == 1:
                                direction = "h"
                            obj_refer = self.ObE.sample_objs_with_line(n=1, len_lims=len_lims, 
                                                                  thickness=thickness, 
                                                                  direction=direction)[0]
                            placement_result = test_canvas.placement(obj_refer, consider_tag=False) # place old obj with free pos
                            if placement_result == -1:
                                break
                        elif node_right.startswith("reactangle"):
                            rect_spec = node_right.split("_")[-1]
                            rect_spec = ast.literal_eval(rect_spec)
                            if rect_spec[0] != -1:
                                w_lims = [rect_spec[1], rect_spec[1]]
                            else:
                                w_lims = [5, test_canvas.init_canvas.shape[1]]
                            if rect_spec[1] != -1:
                                h_lims = [rect_spec[0], rect_spec[0]]
                            else:
                                h_lims = [5, test_canvas.init_canvas.shape[0]]    
                            obj_refer = self.ObE.sample_objs_with_reactangle(n=1, w_lims=w_lims, h_lims=h_lims, 
                                                                        thickness=1, rainbow_prob=0.2)
                            placement_result = test_canvas.placement(obj_refer, consider_tag=False) # place old obj with free pos
                            if placement_result == -1:
                                break
                        elif node_right.startswith("enclosure"):
                            rect_spec = node_right.split("_")[-1]
                            rect_spec = ast.literal_eval(rect_spec)
                            if rect_spec[0] != -1:
                                w_lims = [rect_spec[1], rect_spec[1]]
                            else:
                                w_lims = [3, test_canvas.init_canvas.shape[1]]
                            if rect_spec[1] != -1:
                                h_lims = [rect_spec[0], rect_spec[0]]
                            else:
                                h_lims = [3, test_canvas.init_canvas.shape[0]]
                            if rect_spec[-1] == -1:
                                gravities = [True, False]
                                gravity = random.sample(gravities, k=1)[0]
                            elif rect_spec[-1] == 0:
                                gravity = False
                            else: 
                                gravity = True
                            obj_refer = self.ObE.sample_objs_with_enclosure(n=1, w_lims=w_lims, h_lims=h_lims, 
                                                                       thickness=1, rainbow_prob=0.1, 
                                                                       gravity=gravity)[0]
                            placement_result = test_canvas.placement(obj_refer, consider_tag=False) # place old obj with free pos
                            if placement_result == -1:
                                break
                        elif node_right.startswith("boundary_attachment"):
                            # sample the object and move the desired boundary
                            boundary_spec = node_right.split("_")[-1]
                            boundary_spec = ast.literal_eval(boundary_spec)
                            if boundary_spec[0] == -1:
                                boundary = random.randint(0,3)
                            else:
                                boundary = boundary_spec[0]
                            obj_refer = self.ObE.sample_objs(n=1, is_plot=False)[0]
                            canvas_w = test_canvas.init_canvas.shape[0]
                            canvas_h = test_canvas.init_canvas.shape[1]
                            if boundary == 0:
                                random_pos = [i for i in range(0, canvas_w)]
                                random.shuffle(random_pos)
                                for i in random_pos:
                                    placement_result = \
                                        test_canvas.placement_position_fixed(obj_refer, 0, 
                                                                             i)
                                    if placement_result != -1:
                                        break
                                if placement_result == -1:
                                    break
                            elif boundary == 1:
                                random_pos = [i for i in range(0, canvas_h)]
                                random.shuffle(random_pos)
                                for i in random_pos:
                                    placement_result = \
                                        test_canvas.placement_position_fixed(obj_refer, 
                                                                             i, 
                                                                             canvas_w-obj_refer.image_t.shape[1])
                                    if placement_result != -1:
                                        break
                                if placement_result == -1:
                                    break
                            elif boundary == 2:
                                random_pos = [i for i in range(0, canvas_w)]
                                random.shuffle(random_pos)
                                for i in random_pos:
                                    placement_result = \
                                        test_canvas.placement_position_fixed(obj_refer, 
                                                                             canvas_h-obj_refer.image_t.shape[0],
                                                                             i)
                                    if placement_result != -1:
                                        break
                                if placement_result == -1:
                                    break
                            elif boundary == 3:
                                random_pos = [i for i in range(0, canvas_h)]
                                random.shuffle(random_pos)
                                for i in random_pos:
                                    placement_result = \
                                        test_canvas.placement_position_fixed(obj_refer, 
                                                                             i, 
                                                                             0)
                                    if placement_result != -1:
                                        break
                                if placement_result == -1:
                                    break
                            else:
                                placement_result = -1
                                break
                        else:
                            placement_result = -1
                            break # everything else we are not supported!
                        
                        nodes[node_left] = current_id
                        current_id += 1
                        placed_objs.add(node_left)
                    else:
                        if node_right.startswith("color"):
                            # just need to override the color i think
                            new_c = node_right.split("_")[-1]
                            new_c = ast.literal_eval(new_c)
                            new_c = int(new_c[0]) # get the color
                            placement_result = test_canvas.change_obj_color(nodes[node_left], new_c)
                            if placement_result == -1:
                                break
                        elif node_right.startswith("pos"):
                            # just need to override the position i think
                            new_p = node_right.split("_")[-1]
                            new_p = ast.literal_eval(new_p)
                            if len(new_p) != 2:
                                placement_result = -1
                                break
                            placement_result = test_canvas.change_obj_pos(nodes[node_left], new_p[0], new_p[1])
                            if placement_result == -1:
                                break
                        elif node_right.startswith("boundary_attachment"):
                            # move and attach to the desired boundary
                            boundary_spec = node_right.split("_")[-1]
                            boundary_spec = ast.literal_eval(boundary_spec)
                            if boundary_spec[0] == -1:
                                boundary = random.randint(0,3)
                            else:
                                boundary = boundary_spec[0]
                            canvas_h = test_canvas.init_canvas.shape[0]
                            canvas_w = test_canvas.init_canvas.shape[1]
                            if boundary == 0:
                                random_pos = [i for i in range(0, canvas_w)]
                                random.shuffle(random_pos)
                                for i in random_pos:
                                    placement_result = \
                                        test_canvas.change_obj_pos(nodes[node_left], 0, 
                                                                   i)
                                    if placement_result != -1:
                                        break
                                if placement_result == -1:
                                    break
                            elif boundary == 1:
                                random_pos = [i for i in range(0, canvas_h)]
                                random.shuffle(random_pos)
                                for i in random_pos:
                                    placement_result = \
                                        test_canvas.change_obj_pos(nodes[node_left], 
                                                                   i, 
                                                                   canvas_w-obj_refer.image_t.shape[1])
                                    if placement_result != -1:
                                        break
                                if placement_result == -1:
                                    break
                            elif boundary == 2:
                                random_pos = [i for i in range(0, canvas_w)]
                                random.shuffle(random_pos)
                                for i in random_pos:
                                    placement_result = \
                                        test_canvas.change_obj_pos(nodes[node_left], 
                                                                   canvas_h-obj_refer.image_t.shape[0],
                                                                   i)
                                    if placement_result != -1:
                                        break
                                if placement_result == -1:
                                    break
                            elif boundary == 3:
                                random_pos = [i for i in range(0, canvas_h)]
                                random.shuffle(random_pos)
                                for i in random_pos:
                                    placement_result = \
                                        test_canvas.change_obj_pos(nodes[node_left], 
                                                                   i, 
                                                                   0)
                                    if placement_result != -1:
                                        break
                                if placement_result == -1:
                                    break
                            else:
                                placement_result = -1
                                break
                        else:
                            placement_result == -1 # we currently dont support this DSL
                            break
                    continue

                # check number of new nodes in the proposal
                # some relations require two new nodes come with it
                # some onlt require one

                new_node_count = 0
                if node_left not in placed_objs:
                    new_node_count += 1
                if node_right not in placed_objs:
                    new_node_count += 1
                if new_node_count == 0:
                    raise ValueError("Placement step failed! Check your relation map!")

                if new_node_count == 2:
                    if rel_n == "IsInside":
                        # this is to place the object inside referring to the outside object
                        out_obj = self.ObE.sample_objs_with_reactangle(n=1, thickness=1, rainbow_prob=0.3, 
                                                                  w_lims=[6,8], h_lims=[6,8])[0] 
                        placement_result = test_canvas.placement(out_obj)
                        if placement_result == -1:
                            break
                        nodes[node_right] = current_id
                        current_id += 1
                        in_obj = self.ObE.sample_objs_by_bound_area(n=1, rainbow_prob=0.2, 
                                                               w_lim=3, h_lim=3)[0]
                        placement_result = test_canvas.placement(in_obj, to_relate_objs=[nodes[node_right]], placement_rule="IsInside")
                        if placement_result == -1:
                            break
                        nodes[node_left] = current_id
                        current_id += 1

                    elif rel_n == "SameAll":
                        obj_new = self.ObE.sample_objs(n=1, is_plot=False)[0]
                        obj_new_copy = copy.deepcopy(obj_new)
                        placement_result = test_canvas.placement(obj_new)
                        if placement_result == -1:
                            break
                        nodes[node_left] = current_id
                        current_id += 1
                        placement_result = test_canvas.placement(obj_new_copy)
                        if placement_result == -1:
                            break
                        nodes[node_right] = current_id
                        current_id += 1
                    elif rel_n == "SameRow":
                        obj_anchor = self.ObE.sample_objs(n=1, is_plot=False)[0]
                        placement_result = test_canvas.placement(obj_anchor)
                        if placement_result == -1:
                            break
                        nodes[node_left] = current_id
                        current_id += 1
                        obj_refer = self.ObE.sample_objs_by_fixed_height(n=1, rainbow_prob=0.2, 
                                                                    height=obj_anchor.image_t.shape[0], w_lim=5)[0]
                        placement_result = test_canvas.placement(obj_refer, to_relate_objs=[nodes[node_left]], placement_rule=rel_n)
                        if placement_result == -1:
                            break
                        nodes[node_right] = current_id
                        current_id += 1
                    elif rel_n == "SameCol":
                        obj_anchor = self.ObE.sample_objs(n=1, is_plot=False)[0]
                        placement_result = test_canvas.placement(obj_anchor)
                        if placement_result == -1:
                            break
                        nodes[node_left] = current_id
                        current_id += 1
                        obj_refer = self.ObE.sample_objs_by_fixed_width(n=1, rainbow_prob=0.2, 
                                                                   width=obj_anchor.image_t.shape[1], h_lim=5)[0]
                        placement_result = test_canvas.placement(obj_refer, to_relate_objs=[nodes[node_left]], placement_rule=rel_n)
                        if placement_result == -1:
                            break
                        nodes[node_right] = current_id
                        current_id += 1
                    elif rel_n == "IsTouch":
                        obj_anchor = self.ObE.sample_objs(n=1, is_plot=False)[0]
                        placement_result = test_canvas.placement(obj_anchor)
                        if placement_result == -1:
                            break
                        nodes[node_left] = current_id
                        current_id += 1
                        obj_refer = self.ObE.sample_objs_by_bound_area(n=1, rainbow_prob=0.2, 
                                                                  w_lim=3, h_lim=3)[0]
                        placement_result = test_canvas.placement(obj_refer, to_relate_objs=[nodes[node_left]], placement_rule=rel_n)
                        if placement_result == -1:
                            break
                        nodes[node_right] = current_id
                        current_id += 1

                    elif rel_n == "SameShape":
                        obj_anchor = self.ObE.sample_objs(n=1, is_plot=False)[0]
                        placement_result = test_canvas.placement(obj_anchor)
                        if placement_result == -1:
                            break
                        nodes[node_left] = current_id
                        current_id += 1
                        obj_refer = self.ObE.random_color(obj_anchor)
                        placement_result = test_canvas.placement(obj_refer, placement_rule="SameShape", consider_tag=False) # place old obj with free pos
                        if placement_result == -1:
                            break
                        nodes[node_right] = current_id
                        current_id += 1

                    elif rel_n == "SameColor":
                        obj_anchor = self.ObE.sample_objs(n=1, is_plot=False)[0]
                        placement_result = test_canvas.placement(obj_anchor)
                        if placement_result == -1:
                            break
                        nodes[node_left] = current_id
                        current_id += 1
                        new_c = test_canvas.unify_color(nodes[node_left])
                        # random get an object
                        obj_refer = self.ObE.sample_objs(n=1, is_plot=False)[0]
                        obj_refer = self.ObE.fix_color(obj_refer, new_color=new_c)
                        placement_result = test_canvas.placement(obj_refer, placement_rule="SameColor", consider_tag=False) # place old obj with free pos
                        if placement_result == -1:
                            break
                        nodes[node_right] = current_id
                        current_id += 1
                    placed_objs.add(node_left)
                    placed_objs.add(node_right)
                elif new_node_count == 1:
                    # we only need to place the new obj
                    node_new = node_left if node_left not in placed_objs else node_right
                    node_old = node_left if node_new == node_right else node_right
                    obj_old = test_canvas.get_obj(nodes[node_old])
                    if rel_n == "IsInside":
                        if node_new == node_left:
                            # this is unlikely to succeed as the outer obj is randomly sampled
                            # from our object pool.
                            # the new obj is the inside obj
                            in_obj = self.ObE.sample_objs_by_bound_area(n=1, rainbow_prob=0.2, 
                                                                   w_lim=3, h_lim=3)[0]
                            placement_result = test_canvas.placement(in_obj, 
                                                                     to_relate_objs=[nodes[node_old]], 
                                                                     placement_rule="IsInside")
                        else:
                            # the new obj is the outside obj
                            # this is to place the object inside referring to the outside object
                            out_obj = self.ObE.sample_objs_with_reactangle(n=1, thickness=1, rainbow_prob=0.3, 
                                                                      w_lims=[8,10], h_lims=[8,10])[0] 
                            placement_result = test_canvas.placement(out_obj, 
                                                                     to_relate_objs=[nodes[node_old]], 
                                                                     placement_rule="IsOutside")
                        if placement_result == -1:
                            break
                    elif rel_n == "SameAll":
                        placement_result = test_canvas.placement(obj_old, placement_rule="SameAll", consider_tag=False) # place old obj with free pos
                        if placement_result == -1:
                            break
                    elif rel_n == "SameRow":
                        # we need to actually make sure height is the same
                        height_old = obj_old.image_t.shape[0]
                        in_obj = self.ObE.sample_objs_by_fixed_height(n=1, rainbow_prob=0.2, height=height_old, w_lim=5)[0]
                        placement_result = test_canvas.placement(in_obj, to_relate_objs=[nodes[node_old]], placement_rule=rel_n)
                    elif rel_n == "SameCol":
                        # we need to actually make sure width
                        width_old = obj_old.image_t.shape[1]
                        in_obj = self.ObE.sample_objs_by_fixed_width(n=1, rainbow_prob=0.2, width=width_old, h_lim=5)[0]
                        placement_result = test_canvas.placement(in_obj, to_relate_objs=[nodes[node_old]], placement_rule=rel_n)
                    elif rel_n == "IsTouch":
                        in_obj = self.ObE.sample_objs_by_bound_area(n=1, rainbow_prob=0.2, 
                                                               w_lim=3, h_lim=3)[0]
                        placement_result = test_canvas.placement(in_obj, to_relate_objs=[nodes[node_old]], placement_rule=rel_n)
                        if placement_result == -1:
                            break
                    elif rel_n == "SameShape":
                        obj_new = self.ObE.random_color(obj_old)
                        placement_result = test_canvas.placement(obj_new, placement_rule="SameShape", consider_tag=False) # place old obj with free pos
                        if placement_result == -1:
                            break
                    elif rel_n == "SameColor":
                        new_c = test_canvas.unify_color(nodes[node_old])
                        # random get an object
                        obj_new = self.ObE.sample_objs(n=1, is_plot=False)[0]
                        obj_new = self.ObE.fix_color(obj_new, new_color=new_c)
                        placement_result = test_canvas.placement(obj_new, placement_rule="SameColor", consider_tag=False) # place old obj with free pos
                        if placement_result == -1:
                            break
                    nodes[node_new] = current_id
                    current_id += 1
                    placed_objs.add(node_new)

        if placement_result == -1:
            return -1
                    
        for i in range(self.noise_level):
            # try to add one noise
            noise_obj = self.ObE.sample_objs(n=1, is_plot=False)[0]
            test_canvas.placement(noise_obj, consider_tag=False)
            
        if is_plot:
            test_canvas.render()
            
        return test_canvas.repr_as_dict(nodes, edges)

    def sample_single_task_canvas(self, retry=5, is_plot=True):
        edges = self.sample_single_core_edges()
        print(self.sample_single_canvas_by_core_edges(edges, retry=retry, is_plot=is_plot))
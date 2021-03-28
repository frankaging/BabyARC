import torch
import matplotlib.pylab as plt
# Baby-ARC related imports
try:
    from .constants import *
    from .objects import *
except:
    from constants import *
    from objects import *
    
# general util functions like plots, helpers to manipulate matrics
def remove_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def get_indices(tensor, pos=None, includes_neighbor=False, includes_self=True):
    """Get the indices of nonzero elements of an image.

    Args:
        tensor: 2D or 3D tensor. If 3D, it must have the shape of [C, H, W] where C is the number of channels.
        pos: position of the upper-left corner pixel of the tensor in the larger image. If None, will default as (0, 0).
        includes_neighbor: whether to include indices of neighbors (up, down, left, right).
        includes_self: if includes_neighbor is True, whether to include its own indices.

    Returns:
        indices: a list of indices satisfying the specification.
    """
    mask = tensor > 0
    if len(mask.shape) == 3:
        mask = mask.any(0)
    pos_add = (int(pos[0]), int(pos[1]))  if pos is not None else (0, 0)
    indices = []
    self_indices = []
    for i, j in torch.stack(torch.where(mask)).T:
        i, j = int(i) + pos_add[0], int(j) + pos_add[1]
        self_indices.append((i, j))
        if includes_neighbor:
            indices.append((i + 1, j))
            indices.append((i - 1, j))
            indices.append((i, j + 1))
            indices.append((i, j - 1))
    if includes_neighbor:
        if not includes_self:
            indices = list(set(indices).difference(set(self_indices)))
        else:
            indices = remove_duplicates(indices)
    else:
        indices = self_indices
    return indices

def plot_with_boundary(image, plt):
    im = plt.imshow(image, interpolation='none', vmin=0, vmax=1, aspect='equal');
    height, width = np.array(image).shape[:2]
    ax = plt.gca();

    # Major ticks
    ax.set_xticks(np.arange(0, width, 1));
    ax.set_yticks(np.arange(0, height, 1));

    # Labels for major ticks
    ax.set_xticklabels(np.arange(1, width + 1, 1));
    ax.set_yticklabels(np.arange(1, height + 1, 1));

    # Minor ticks
    ax.set_xticks(np.arange(-.5, width, 1), minor=True);
    ax.set_yticks(np.arange(-.5, height, 1), minor=True);

    # Gridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
    
    plt.xticks([])
    plt.yticks([])

def get_object_position_tags(obj_pos_t, root_shape_t):
    position_tags = []
    # using the first two bit of determine whether attach to left or up
    if obj_pos_t[0] == 0:
        position_tags.append("upper")
    if obj_pos_t[1] == 0:
        position_tags.append("left")
    if (obj_pos_t[0]+obj_pos_t[2]) == root_shape_t[0]:
        position_tags.append("lower")
    if (obj_pos_t[1]+obj_pos_t[3]) == root_shape_t[1]:
        position_tags.append("right")
    return position_tags

def randint_exclude(l, u, e):
    r = e[0]
    while r in e:
        r = random.randint(l, u)
    return r

def single_task_obj_parser(task_id):
    """
    return a list of objects contained in this task.
    """
    task_objs = []
    task_id += ".json"
    (inputs, _), (_, _) = load_task(task_id, isplot=False)
    # parse objects in 3 ways
    inputs_graph = parse_obj(inputs)
    for i in range(len(inputs_graph[0])):
        root_shape_t = inputs_graph[0][i].get_node_value("Image").shape
        for obj_n in inputs_graph[0][i].objs:
            obj_img_t = inputs_graph[0][i].get_node_value(obj_n)
            obj_pos_t = inputs_graph[0][i].get_node_value(obj_n.split(":")[0]+"^pos:Pos")
            obj_position_tags = get_object_position_tags(obj_pos_t, root_shape_t)
            obj_fmt = Object(obj_img_t, obj_position_tags)
            task_objs.append(obj_fmt)
            
    inputs_graph = parse_obj(inputs, is_colordiff=False)
    for i in range(len(inputs_graph[0])):
        root_shape_t = inputs_graph[0][i].get_node_value("Image").shape
        for obj_n in inputs_graph[0][i].objs:
            obj_img_t = inputs_graph[0][i].get_node_value(obj_n)
            obj_pos_t = inputs_graph[0][i].get_node_value(obj_n.split(":")[0]+"^pos:Pos")
            obj_position_tags = get_object_position_tags(obj_pos_t, root_shape_t)
            obj_fmt = Object(obj_img_t, obj_position_tags)
            task_objs.append(obj_fmt)
            
    inputs_graph = parse_obj(inputs, is_diag=False)
    for i in range(len(inputs_graph[0])):
        root_shape_t = inputs_graph[0][i].get_node_value("Image").shape
        for obj_n in inputs_graph[0][i].objs:
            obj_img_t = inputs_graph[0][i].get_node_value(obj_n)
            obj_pos_t = inputs_graph[0][i].get_node_value(obj_n.split(":")[0]+"^pos:Pos")
            obj_position_tags = get_object_position_tags(obj_pos_t, root_shape_t)
            obj_fmt = Object(obj_img_t, obj_position_tags)
            task_objs.append(obj_fmt)
    return task_objs

# relation parser as we need for multihop reasonings
def SameShape(image1, pos1, image2, pos2):
    if np.prod(image1.shape) == 0:
        return False
    if np.prod(image2.shape) == 0:
        return False
    if image1.shape != image2.shape:
        return False
    else:
        return (image1.bool() == image2.bool()).all()

def SameColor(image1, pos1, image2, pos2):
    color1 = -1 if len(image1.unique()) != 1 else image1.unique()[0]
    color2 = -1 if len(image2.unique()) != 1 else image2.unique()[0]
    if color1 == -1 or color2 == -1:
        return False
    else:
        return color1 == color2

def SameAll(image1, pos1, image2, pos2):
    if np.prod(image1.shape) == 0:
        return False
    if np.prod(image2.shape) == 0:
        return False
    if image1.shape != image2.shape:
        return False
    else:
        return (image1 == image2).all()

def SameRow(image1, pos1, image2, pos2):
    pos1 = (pos1[0], pos1[1], image1.shape[0], image1.shape[1])
    pos2 = (pos2[0], pos2[1], image2.shape[0], image2.shape[1])
    if pos1[0] == pos2[0] and pos1[2] == pos2[2]:
        return True
    else:
        return False

def SameCol(image1, pos1, image2, pos2):
    pos1 = (pos1[0], pos1[1], image1.shape[0], image1.shape[1])
    pos2 = (pos2[0], pos2[1], image2.shape[0], image2.shape[1])
    if pos1[1] == pos2[1] and pos1[3] == pos2[3]:
        return True
    else:
        return False

def IsInside(image1, pos1, image2, pos2):
    """Whether obj1 is inside obj2."""
    pos1 = (pos1[0], pos1[1], image1.shape[0], image1.shape[1])
    pos2 = (pos2[0], pos2[1], image2.shape[0], image2.shape[1])
    if pos1[0] > pos2[0] and pos1[1] > pos2[1] and pos1[0] + pos1[2] < pos2[0] + pos2[2] and pos1[1] + pos1[3] < pos2[1] + pos2[3]:
        image2_patch = image2[int(pos1[0] - pos2[0]): int(pos1[0] + pos1[2] - pos2[0]), 
                              int(pos1[1] - pos2[1]): int(pos1[1] + pos1[3] - pos2[1])]
        overlap = (image1 != 0) & (image2_patch != 0)
        if overlap.any():
            return False
        else:
            return True
    else:
        return False
    
def IsTouch(image1, pos1, image2, pos2):
    """Whether the "obj"'s leftmost/rightmost/upmost/downmost part touches any other pixels (up, down, left, right) or boundary in the "image"."""
    pos1 = (pos1[0], pos1[1], image1.shape[0], image1.shape[1])
    pos2 = (pos2[0], pos2[1], image2.shape[0], image2.shape[1])
    obj_indices = get_indices(
        image1,
        pos1,
        includes_self=False,
        includes_neighbor=True,
    )
    obj2_indices = get_indices(image2, pos2)
    is_torch = len(set(obj_indices).intersection(set(obj2_indices))) > 0
    return is_torch
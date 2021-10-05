# BabyARC

Baby Abstract Reasoning Corpus (BabyARC) dataset engine, for generating grid-world-based abstract reasoning tasks on a large scale. BabyARC's name is inspired by [the Abstract Reasoning Corpus (ARC)](https://github.com/fchollet/ARC). The goal of this dataset engine is to generate concept, relation and reasoning pretraining data source for solving the ARC corpus.

## Contents

* [Example](#example)
* [Usage](#usage)

## Example

Here, we show concept and relation canvases can be generated using BabyARC:

<img src="https://i.ibb.co/kMWKvv4/Baby-ARC-Examples.png" width="800">

Here, we show some simplified abstract reasoning tasks can be generated using BabyARC:

<img src="https://i.ibb.co/yFGvpM9/Baby-ARC-Task.png" width="800">

## Usage

```python
# generating concepts and relation canvases.
from code.dataset.dataset import *
demo_dataset = \
    BabyARCDataset(pretrained_obj_cache=os.path.join(get_root_dir(), 'concept_env/datasets/arc_objs.pt'),
                   save_directory="./BabyARCDataset/", 
                   object_limit=1, noise_level=0, 
                   canvas_size=8) # canvas makes w=h canvas

# Inplace object placement.
canvas_dict = demo_dataset.sample_single_canvas_by_core_edges(
    OrderedDict(
        [(('obj_0', 'obj_1'), 'SameColor'),
         (('obj_0', 'obj_2'), 'IsTouch')]
    ), 
    color_avail=[1,2,3],
    rainbow_prob=0.0,
    allow_connect=True, 
)
```

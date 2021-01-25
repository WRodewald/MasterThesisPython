# MasterThesisPython

This repository contains python code for my master thesis. 
For temporary data, a ``build`` directory is created in the root path.

## VocalSet
The VocaSet used in the thesis needs to be downloaded separately (https://zenodo.org/record/1193957). We use the structure organized by singer. To connect to vocalset, call ``python hsvs/tools/dataset.py --set <path_to_vocalset_root>``. 
The path will be stored in `build/vocalset.json`. 

## Dependencies

``numpy``\
``scipy``\
``soundfile``\
``matplotlib``\
``tensorflow``\
``crepe`` (https://github.com/marl/crepe)\
``tqdm`` (https://github.com/tqdm/tqdm)\
``pandas``\
``packaging``

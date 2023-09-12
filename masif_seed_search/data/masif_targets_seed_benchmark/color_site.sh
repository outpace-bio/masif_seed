#!/bin/bash
masif_root=/root/masif_seed
masif_source=$masif_root/masif/source/
masif_data=$masif_root/masif/data/
export PYTHONPATH=$masif_source:$masif_data/masif_site/
python -W ignore $masif_source/masif_site/masif_site_label_surface.py nn_models.all_feat_3l_seed_benchmark.custom_params $1 $2

masif_root=/root/masif_seed
masif_source=$masif_root/masif/source/
masif_seed_source=../../source/
masif_data=$masif_root/masif/data/
export PYTHONPATH=$PYTHONPATH:$masif_source:
python -W ignore $masif_seed_source/precompute_evaluation_features.py training_data_12A_seed_benchmark/ $1

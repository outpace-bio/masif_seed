masif_root=/root/masif_seed
masif_source=$masif_root/source/
masif_data=$masif_root/data/
export PYTHONPATH=$PYTHONPATH:$masif_source:$masif_data/masif_ppi_search/:$masif_data/masif_ppi_search/transformation_training_data/
masif_root=/root/masif_seed
python $masif_source/masif_ppi_search/transformation_training_data/train_evaluation_network.py

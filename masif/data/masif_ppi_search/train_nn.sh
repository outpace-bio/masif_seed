masif_root=/root/masif_seed
masif_source=$masif_root/masif/source/
masif_data=$masif_root/masif/data/
export PYTHONPATH=$PYTHONPATH:$masif_source:$masif_data/masif_ppi_search/
python $masif_source/masif_ppi_search/masif_ppi_search_train.py $1

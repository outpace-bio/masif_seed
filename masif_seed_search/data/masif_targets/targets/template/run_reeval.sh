masif_root=/root/masif_seed
export masif_db_root=/home/gainza/lpdi_fs/masif/
masif_source=$masif_root/source/
masif_matlab=$masif_root/source/matlab_libs/
masif_data=$masif_root/data/
export masif_root
export PYTHONPATH=$PYTHONPATH:$masif_source:`pwd`
#python $masif_source/masif_seed_search/masif_seed_search_nn.py params 4ZQK_A
#python $masif_source/masif_seed_search/masif_seed_search_nn.py params_small_proteins 5JDS_A
python $masif_source/masif_seed_search/masif_seed_search_nn.py params_small_proteins_reeval 4ZQK_A
#python $masif_source/masif_seed_search/masif_seed_search_nn.py params_peptides 4ZQK_A

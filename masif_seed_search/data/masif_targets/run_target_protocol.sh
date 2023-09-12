#!/bin/bash
masif_seed_root=/root/masif_seed
masif_seed_targets=$masif_seed_root/masif_seed_search/data/masif_targets
$masif_seed_targets/data_prepare_one.sh $1
$masif_seed_targets/predict_site.sh $1
$masif_seed_targets/color_site.sh $1
$masif_seed_targets/compute_descriptors.sh $1
echo "Creating running directory targets/$1 "
cp -r $masif_seed_targets/targets/template/ targets/$1
## UNCOMMENT THESE LINES IF YOU WANT D-AMINO ACIDS
# Now compute descriptors and predictions for the mirror image target (d-amino acids)
#./predict_site.sh d$1
#./color_site.sh d$1
#./compute_descriptors.sh d$1
#echo "Creating running directory for the mirror image (d-amino acids): targets/d$1 "
#cp -r targets/template/ targets/d$1


#!/usr/bin/env sh

mkdir -p data

cd data

echo "downloading tiny nerf..."
# wget https://people.eecs.berkeley.edu/~bmild/nerf/tiny_nerf_data.npz # expired
curl -C - https://cseweb.ucsd.edu//~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz -o tiny_nerf_data.npz || \
    echo failed... try https://github.com/bmild/nerf/issues
sha512sum tiny_nerf_data.npz > tiny_nerf_data.npz.sha512

echo
echo
echo "downloading nerf synthetic..."
curl -C - -L -o nerf-synthetic.zip https://www.kaggle.com/api/v1/datasets/download/nguyenhung1903/nerf-synthetic-dataset || \
    echo failed... try https://github.com/bmild/nerf/issues

#curl -C - -L  https://www.kaggle.com/api/v1/datasets/download/alexlwh/nerf-synthetic -o nerf-synthetic.zip || \
#    echo failed... try https://github.com/bmild/nerf/issues
unzip -qn nerf-synthetic.zip
find nerf_synthetic -type f | sort | parallel -k 'sha512sum {}' > nerf_synthetic.sha512

echo
echo
echo "downloading mip-nerf 360..."
curl -C - "https://storage.googleapis.com/gresearch/refraw360/360_v2.zip" -o 360_v2.zip || \
    echo failed... try https://github.com/google-research/multinerf/issues
unzip -qn 360_v2.zip -d 360_v2
find 360_v2 -type f | sort | parallel -k 'sha512sum {}' > 360_v2.sha512


curl -C - "https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip" -o 360_extra_scenes.zip || \
    echo failed... try https://github.com/google-research/multinerf/issues
unzip -qn 360_extra_scenes.zip -d 360_extra_scenes
find 360_extra_scenes -type f | sort | parallel -k 'sha512sum {}' > 360_extra_scenes.sha512

echo
echo
echo verifying integrity...
cd ..
sha512sum -c data.sha512

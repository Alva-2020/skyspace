#!/bin/bash

model="xvector"

if [ $model == "xvector" ]; then
python train.py  --config  xvector.yml

fi

if [ $model == "resnet" ]; then

echo "resnet model"

fi

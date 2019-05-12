#!/bin/bash

mkdir -p datos/optdigits
wget https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra -O datos/optdigits/optdigits.tra
wget https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes -O datos/optdigits/optdigits.tes

mkdir -p datos/airfoil
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat -O datos/airfoil/airfoil_self_noise.dat

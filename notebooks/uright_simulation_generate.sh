#!/usr/bin/env bash

declare -a USER=('user_6' 'user_9' 'user_29' 'user_32' 'user_35')

for i in "${USER[@]}"
do
    echo $i
    cp uright_simulation_user_1.ipynb uright_simulation_$i.ipynb
    sed -i "s/user_1/$i/g" uright_simulation_$i.ipynb
done

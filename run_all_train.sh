#!/bin/bash


nohup python3 train.py --data_path="archimob_sentences.csv" --model_name="archimob" --log_path="./logs/archimob/" --input_alphabet=archimob --drop_last=0 --balance=0 > train_archimob.out 2>&1 &
nohup python3 train.py --data_path="archimob_sentences.csv" --model_name="archimob_balanced" --log_path="./logs/archimob_balanced/" --input_alphabet=archimob --drop_last=0 --balance=1 > train_archimob.balanced.out 2>&1 &
nohup python3 train.py --data_path="archimob_sentences.csv" --model_name="archimob_drop_last" --log_path="./logs/archimob_drop_last/" --input_alphabet=archimob --drop_last=1 > train_archimob.drop_last.out 2>&1 &
nohup python3 train.py --data_path="archimob_sentences.csv" --model_name="archimob_balanced_drop_last" --log_path="./logs/archimob_balanced_drop_last/" --input_alphabet=archimob --drop_last=1 --balance=1 > train_archimob.balanced.drop_last.out 2>&1 &

nohup python3 train.py --data_path="~/sentences_ch_de_numerics.train.csv" --model_name="swissdial" --log_path="./logs/swissdial/" --input_alphabet=swissdial --drop_last=0 --balance=0 > train_swissdial.out 2>&1 &
nohup python3 train.py --data_path="~/sentences_ch_de_numerics.train.csv" --model_name="swissdial_balanced" --log_path="./logs/swissdial_balanced/" --input_alphabet=swissdial --drop_last=0 --balance=1 > train_swissdial.balanced.out 2>&1 &
nohup python3 train.py --data_path="~/sentences_ch_de_numerics.train.csv" --model_name="swissdial_drop_last" --log_path="./logs/swissdial_drop_last/" --input_alphabet=swissdial --drop_last=1 > train_swissdial.drop_last.out 2>&1 &
nohup python3 train.py --data_path="~/sentences_ch_de_numerics.train.csv" --model_name="swissdial_balanced_drop_last" --log_path="./logs/swissdial_balanced_drop_last/" --input_alphabet=swissdial --drop_last=1 --balance=1 > train_swissdial.balanced.drop_last.out 2>&1 &

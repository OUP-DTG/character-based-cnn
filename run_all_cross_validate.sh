#!/bin/bash


nohup python3 train_cross_validate.py --data_path="archimob_sentences.csv" --model_name="archimob" --log_path="./logs/archimob/" --input_alphabet=archimob --kfolds=5 --drop_last=0 --balance=0 > archimob.out 2>&1 &
nohup python3 train_cross_validate.py --data_path="archimob_sentences.csv" --model_name="archimob_balanced" --log_path="./logs/archimob_balanced/" --input_alphabet=archimob --kfolds=5 --drop_last=0 --balance=1 > archimob.balanced.out 2>&1 &
nohup python3 train_cross_validate.py --data_path="archimob_sentences.csv" --model_name="archimob_drop_last" --log_path="./logs/archimob_drop_last/" --input_alphabet=archimob --kfolds=5 --drop_last=1 > archimob.drop_last.out 2>&1 &
nohup python3 train_cross_validate.py --data_path="archimob_sentences.csv" --model_name="archimob_balanced_drop_last" --log_path="./logs/archimob_balanced_drop_last/" --input_alphabet=archimob --kfolds=5 --drop_last=1 --balance=1 > archimob.balanced.drop_last.out 2>&1 &

nohup python3 train_cross_validate.py --data_path="~/sentences_ch_de_numerics.train.csv" --model_name="swissdial" --log_path="./logs/swissdial/" --input_alphabet=swissdial --kfolds=5 --drop_last=0 --balance=0 > swissdial.out 2>&1 &
nohup python3 train_cross_validate.py --data_path="~/sentences_ch_de_numerics.train.csv" --model_name="swissdial_balanced" --log_path="./logs/swissdial_balanced/" --input_alphabet=swissdial --kfolds=5 --drop_last=0 --balance=1 > swissdial.balanced.out 2>&1 &
nohup python3 train_cross_validate.py --data_path="~/sentences_ch_de_numerics.train.csv" --model_name="swissdial_drop_last" --log_path="./logs/swissdial_drop_last/" --input_alphabet=swissdial --kfolds=5 --drop_last=1 > swissdial.drop_last.out 2>&1 &
nohup python3 train_cross_validate.py --data_path="~/sentences_ch_de_numerics.train.csv" --model_name="swissdial_balanced_drop_last" --log_path="./logs/swissdial_balanced_drop_last/" --input_alphabet=swissdial --kfolds=5 --drop_last=1 --balance=1 > swissdial.balanced.drop_last.out 2>&1 &

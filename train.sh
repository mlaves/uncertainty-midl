#!/bin/sh

rm -rf __pycache__/
BS=128
python -u train.py --bs ${BS} --epoch 5 --model baseline      | tee `date '+%Y-%m-%d_%H-%M-%S'`_baseline.log
python -u train.py --bs ${BS} --epoch 5 --model bayesian1     | tee `date '+%Y-%m-%d_%H-%M-%S'`_bayesian1.log
python -u train.py --bs ${BS} --epoch 8 --model bayesian2     | tee `date '+%Y-%m-%d_%H-%M-%S'`_bayesian2.log
python -u train.py --bs ${BS} --epoch 5 --model probabilistic | tee `date '+%Y-%m-%d_%H-%M-%S'`_probabilistic.log

#!/bin/sh

set -ex

# baseline
python answer.py --name exp001_baseline --seed 42
python answer.py --name exp001_baseline --seed 43
python answer.py --name exp001_baseline --seed 44
python answer.py --name exp001_baseline --seed 45
python answer.py --name exp001_baseline --seed 46

# sota
python answer.py --name exp002_sota --seed 42 --use_sota
python answer.py --name exp002_sota --seed 43 --use_sota
python answer.py --name exp002_sota --seed 44 --use_sota
python answer.py --name exp002_sota --seed 45 --use_sota
python answer.py --name exp002_sota --seed 46 --use_sota
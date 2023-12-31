## PDTE

This is an implementation of the protocols in paper [Constant-Round Private Decision Tree Evaluation for Secret Shared Data](https://eprint.iacr.org/2023/1307),  will appear at PETS 2024.

## Pre-requirments 

`pip install numpy sycret`

`pip install -U scikit-learn`

`python3 -m pip install tno.mpc.communication`

## prepare pre-trained model

`cd ../data/train`

`python3 train.py`

<!--## Structure of the code files

 * function secret sharing implementation: fss.py 

 * func -->

## Configuration for benchmarking

+ Configure IP addresses in `common/constants.py` file. Change
`SERVER_IPS = ["127.0.0.1", "127.0.0.1", "127.0.0.1"]` 
to enable a LAN setting, or change them to your own  cloud-server IP addresses for a real WAN setting.

+ In `benchmarkOption.py` file, change the test index id `BENCHMARK_CHOICE_DATASET_IDX`,  which indicates the index whithin dataset `["wine", "linnerud", "cancer", "digits-10", "digits-12", "digits-15","diabets-18","diabets-20"]`, as well as other benchmarking options defined you need.


## How to run the benchmarking

First, we need to prepare the offline phase, i.e., subsequentially run

`python3 offline.py 0`
`python3 offline.py 1`
`python3 offline.py 2`,

which will output the pre-processing data required for online evaluation.

Then, subsequentially run

`python3 offline.py 0`
`python3 offline.py 1`
`python3 offline.py 2`.
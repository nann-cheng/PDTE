## PDTE

This is an implementation of the protocols in paper [Constant-Round Private Decision Tree Evaluation for Secret Shared Data](https://eprint.iacr.org/2023/1307),  will appear at PETS 2024. This protocol involves three servers running the private decision tree algorithm provided by the **decision tree model** and a **sample vector**.


## Structure of the code files

 * `common/`: 
    * `completeBinaryTree.py`: A realization of complete binary tree node.
    * `constants.py`: Constant values used for network communcation, randomness generation, etc.
    * `helper.py`: Common helper functions used through different files.
    function secret sharing implementation: fss.py 
 * `model/`: A repo that contains a script `train.py` that could generate the output decision tree model with file format `*.model`, which are already provided as well here. These fiels serve as input to the desired protocol for private evaluation.
 * `parties/`: This repo realizes the behaviours of servers in the **featSelect** phase and **pathEval** phase as described in the paper.
 * `benchmarkOption.py`: In this file, adjustation can be made for receiving benchmarking result.
 * `bTree.py`: A realization of binary tree data structure.
 * `dealer.py`: A realization of the trustful thrid party who deals the distributed correlated randomness data in the offline phase.
 * `fss.py`: The function secret sharing implementation used for performing secure comparison.
 * `player.py`: A implementation of the offline/online behaviors of each server.
 * `server.py`: The script that takes in an integer, indicating a party id, and evaluates the offline+online phase performance.

## Pre-requirments 

Run script `./setupEnv.sh`, which installs all the required python libraries.

## The prepared pre-trained model

The required pre-trained decision tree model can be found at repo `model/`, which are all files come with `*.model` format. These files are obtained from running following commands.

`cd model/`
`python3 train.py`

## Configuration for benchmarking

+ Configure IP addresses in `benchmarkOptions.py` file. Set
`SERVER_IPS = ["127.0.0.1", "127.0.0.1", "127.0.0.1"]` 
to enable a LAN setting, or adapt them to your own cloud-server IP addresses for a real WAN setting.

+ In `benchmarkOption.py` file, change the test index id `BENCHMARK_CHOICE_DATASET_IDX` from 0 to 6,  which indicates the index whithin dataset `["wine", "linnerud", "cancer", "digits-10", "digits-12", "digits-15","diabets-18",]`.

## How to run the benchmarking

In `server.py`, we have integrated the offline and online sub-protocols altogether, simply by running

`python3 server.py 0`
`python3 server.py 1`
`python3 server.py 2`,

we are able to have a full performance test over all offline/online sub-protocols.

## How to interpret the output from the benchmarking 

After running
`python3 server.py 0`
`python3 server.py 1`
`python3 server.py 2`,

it's expected to have all the performance report regarding

 - Offline-phase communication cost: this can be found in the terminal output of `python3 server.py 1`, where we can find somewhere in the output with
    - `*********Offline FeatSelect preparation communication cost's: xx bytes!*********`.
    - `*********Offline PathEval preparation communication cost's: xx bytes!*********`.
    - `*********Offline Compare preparation communication cost's: xx bytes!*********`.

    this directly gives us the offline-phase communication cost.

 - Online-phase communication/computation cost: these numbers appear at every terminal output of `python3 server.py 0/1/2`', where we can find somewhere in the output with
    - `******************Online FeatSelect communication cost's: xx bytes!*********`.
    - `******************Online FeatureSelect time's: xx s!*********`.

    - `*********Online Compare communication cost's: xx bytes!*********`.
    - `*********Online compare computation time's: xx s!*********`.

    - `*********Online PathEval communication cost's: xx bytes!*********`.
    - `*********Online PathEval computation time's: xx s!*********`.

    - `*********Total online computation time's: xx s!*********`.
    
    However, it's required to sum up all partial cost together to get the average cost regarding of computation time or communication volume cost with measurement above.

Additionally we can change the value of `BENCHMARK_CHOICE_DATASET_IDX` from 0 to 6, such that we could test over the corresponding indexed item in dataset `["wine", "linnerud", "cancer", "digits-10", "digits-12", "digits-15","diabets-18",]`.
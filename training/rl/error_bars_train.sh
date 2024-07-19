#!/bin/bash
# usage: error_bars_train.sh <path to rlxf main.py> <path to config> <number of runs>
if [ "$#" -ne 3 ]; then
    echo "Usage: error_bars_train.sh <path to rlxf main.py> <path to config> <number of runs>"
    exit 1
fi

RLXF=$1
CONFIG=$(realpath $2)
N=$3

pushd $(dirname $RLXF)

for i in $(seq 1 $N); do
    # modify config json file to include seed run index. 
    # use jq to modify json file
    TMP_CONFIG=$(mktemp)
    cp $CONFIG $TMP_CONFIG
    # change: "save_location": "<path to out>" to "save_location": "<path to out>/run_<i>"
    jq --arg i $i '.save_location = "\(.save_location)/run_\($i)"' $TMP_CONFIG > $TMP_CONFIG.tmp
    # if present, change "wandb"/"name": "<name>" to "wandb"/"name": "<name>_run_<i>"
    jq --arg i $i 'if .wandb.name then .wandb.name = "\(.wandb.name)_run_\($i)" else . end' $TMP_CONFIG.tmp > $TMP_CONFIG
    cat $TMP_CONFIG
    SEED="$i" python3 main.py --config $TMP_CONFIG
done

popd

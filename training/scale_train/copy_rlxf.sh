#!/bin/bash

MODELS_PATH=${MODELS_PATH:-"~/src/models/"}

if [ ! -d $MODELS_PATH ]; then
    echo "Error: $MODELS_PATH does not exist. Set MODELS_PATH to the correct path."
    exit 1
fi

pushd "$(realpath "$(dirname "$0")")" > /dev/null

rm -fr ../rl/rlxf
cp -r $MODELS_PATH/rlxf ../rl/rlxf
rm -fr ../rl/rlxf/.git ../rl/rlxf/wandb ../rl/rlxf/__pycache__

popd

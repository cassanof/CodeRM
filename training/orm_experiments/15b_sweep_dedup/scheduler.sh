# usage: ./scheduler.sh <from> <to>
# This script runs a sweep over all configurations of the model with the given range of hyperparameters.

if [ "$#" -ne 2 ]; then
  echo "Usage: ./scheduler.sh <from> <to>"
  exit 1
fi

FROM=$1
TO=$2

GRAD_ACCS=("8" "16" "32")
LRS=("1e-5" "2e-5" "4e-5")

echo "Total configurations: ${#LRS[@]} x ${#GRAD_ACCS[@]} = $(( ${#LRS[@]} * ${#GRAD_ACCS[@]} ))"

export WANDB_PROJECT="coderm-sweep-15b"
export WANDB_JOB_TYPE="sweep"

# loop over all configurations
COUNTER=0
for grad_acc in ${GRAD_ACCS[@]}; do
  for lr in ${LRS[@]}; do
    C=$COUNTER
    COUNTER=$((COUNTER+1))
    if [ $C -lt $FROM ]; then
      continue
    fi
    if [ $C -ge $TO ]; then
      break
    fi
    echo "Running: GRAD_ACC=$grad_acc, LR=$lr"
    ./train.sh $grad_acc $lr
  done
done


GRAD_ACCS=("8" "16" "32")
LRS=("1e-5" "2e-5" "4e-5")

echo "Total configurations: ${#LRS[@]} x ${#GRAD_ACCS[@]} = $(( ${#LRS[@]} * ${#GRAD_ACCS[@]} ))"

export WANDB_PROJECT="codeprm-sweep-15b"
export WANDB_JOB_TYPE="sweep"

# loop over all configurations
for grad_acc in ${GRAD_ACCS[@]}; do
  for lr in ${LRS[@]}; do
    echo "Running: GRAD_ACC=$grad_acc, LR=$lr"
    ./train.sh $grad_acc $lr
  done
done


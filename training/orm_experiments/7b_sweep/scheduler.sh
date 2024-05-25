GRAD_ACCS=("2" "4" "8" "16")
LRS=("1e-5" "2e-5" "4e-5" "8e-5" "1e-4")

echo "Total configurations: ${#LRS[@]} x ${#GRAD_ACCS[@]} = $(( ${#LRS[@]} * ${#GRAD_ACCS[@]} ))"

export WANDB_PROJECT="codeprm"
export WANDB_JOB_TYPE="sweep"

# loop over all configurations
for grad_acc in ${GRAD_ACCS[@]}; do
  for lr in ${LR[@]}; do
    echo "Running: GRAD_ACC=$grad_acc, LR=$lr"
    ./train.sh $grad_acc $lr
  done
done


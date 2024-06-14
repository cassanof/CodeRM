from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
import wandb
import os


class MakeShiftWandbCallback(TrainerCallback):
    def __init__(self):
        self._initialized = False

    def setup(
            self,
            args: TrainingArguments,
            state: TrainerState,
    ):
        trial_name = state.trial_name
        init_args = {}
        if trial_name is not None:
            init_args["name"] = trial_name
            init_args["group"] = args.run_name
        else:
            if not (args.run_name is None or args.run_name == args.output_dir):
                init_args["name"] = args.run_name

        if state.is_local_process_zero:
            config = args.to_dict()
            for key in model_config.to_dict():
                if key not in config:
                    config[key] = model_config.to_dict()[key]

            wandb.init(
                project=os.getenv("WANDB_PROJECT", "rl"),
                **init_args,
                config=config,
            )
            wandb.define_metric("train/global_step")
            wandb.define_metric(
                "*", step_metric="train/global_step", step_sync=True)

    def on_log(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            logs=None,
            **kwargs,
    ):
        if not self._initialized:
            self.setup(args, state)
            self._initialized = True

        if logs is None:
            logs = {}

        if state.is_world_process_zero:
            wandb.log(
                {**logs, "train/global_step": state.global_step})

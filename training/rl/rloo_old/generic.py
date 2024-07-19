from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from torch import nn
from torch.nn import functional as F
import wandb
import os


class MakeShiftWandbCallback(TrainerCallback):
    def __init__(self):
        self._initialized = False

    def setup(
            self,
            args: TrainingArguments,
            state: TrainerState,
            model=None,
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
            if model is not None and hasattr(model, "config") and model.config is not None and hasattr(model.config, "to_dict"):
                model_config = model.config
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
            model=None,
            **kwargs,
    ):
        if not self._initialized:
            self.setup(args, state, model=model)
            self._initialized = True

        if logs is None:
            logs = {}

        if state.is_world_process_zero:
            wandb.log(
                {**logs, "train/global_step": state.global_step})


class SingleLabelAdapterLayer(nn.Module):
    def __init__(self, layer, pos_idx=1):
        super().__init__()
        self.layer = layer
        self.pos_idx = pos_idx

    def forward(self, input, *args, **kwargs):
        logits = self.layer(input, *args, **kwargs)
        logits = F.softmax(logits, dim=-1)
        logits = logits[:, :, self.pos_idx].unsqueeze(-1)
        return logits


def convert_2_label_rm_to_1_label_rm(model, score_field="score", pos_idx=1):
    prev_score = getattr(model, score_field)
    new_score = SingleLabelAdapterLayer(prev_score, pos_idx)
    setattr(model, score_field, new_score)
    model.config.num_labels = 1
    return model

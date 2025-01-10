from dataclasses import dataclass
from datetime import datetime
from distutils.util import strtobool
import logging
import os
import re
from typing import Optional
import torch
import mlflow
from transformers.integrations import MLflowCallback
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import is_liger_kernel_available
from trl import SFTTrainer, TrlParser, ModelConfig, SFTConfig, get_peft_config
from datasets import load_dataset

if is_liger_kernel_available():
    from liger_kernel.transformers import AutoLigerKernelForCausalLM


@dataclass
class ScriptArguments:
    dataset_id_or_path: str
    dataset_splits: str = "train"
    tokenizer_name_or_path: str = None
    spectrum_config_path: Optional[str] = None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)


def get_checkpoint(training_args: SFTConfig):
    last_checkpoint = False
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def setup_model_for_spectrum(model, spectrum_config_path):
    unfrozen_parameters = []
    with open(spectrum_config_path, "r") as fin:
        yaml_parameters = fin.read()

    # get the unfrozen parameters from the yaml file
    for line in yaml_parameters.splitlines():
        if line.startswith("- "):
            unfrozen_parameters.append(line.split("- ")[1])

    # freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # unfreeze Spectrum parameters
    for name, param in model.named_parameters():
        if any(re.match(unfrozen_param, name) for unfrozen_param in unfrozen_parameters):
            param.requires_grad = True
    
    return model


def train_function(model_args: ModelConfig, script_args: ScriptArguments, training_args: SFTConfig):
    """Main training function."""

    if script_args.dataset_id_or_path.endswith('.json'):
        train_dataset = load_dataset('json', data_files=script_args.dataset_id_or_path, split='train')
    else:
        train_dataset = load_dataset(script_args.dataset_id_or_path, split=script_args.dataset_splits)
    
    if training_args.local_rank == 0 or training_args.process_index == 0:     
        logger.info(f'Loaded dataset with {len(train_dataset)} samples and the following features: {train_dataset.features}')
        if os.environ.get('TEST_MODE') == 'true':
            # For development, only take 20% of the dataset.
            logger.info(f'Loading only 30% of the dataset for development.')
            train_dataset = train_dataset.select(range(int(len(train_dataset) * 0.3)))

    tokenizer = AutoTokenizer.from_pretrained(
        script_args.tokenizer_name_or_path if script_args.tokenizer_name_or_path else model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=(
            model_args.torch_dtype
            if model_args.torch_dtype in ['auto', None]
            else getattr(torch, model_args.torch_dtype)
        ),
        use_cache=False if training_args.gradient_checkpointing else True,
        # If using DeepSpeed through accelerate, you might set low_cpu_mem_usage to None
        low_cpu_mem_usage=(
            True
            if not strtobool(os.environ.get("ACCELERATE_USE_DEEPSPEED", "false"))
            else None
        ),
    )

    if model_args.load_in_4bit:
        model_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=model_kwargs['torch_dtype'],
            bnb_4bit_quant_storage=model_kwargs['torch_dtype'],
        )

    if model_args.use_peft:
        peft_config = get_peft_config(model_args)
    else:
        peft_config = None

    if training_args.use_liger and is_liger_kernel_available():
        model = AutoLigerKernelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    if script_args.spectrum_config_path:
        model = setup_model_for_spectrum(model, script_args.spectrum_config_path)

    mlflow_callback = MLflowCallback()

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        callbacks=[mlflow_callback],
    )

    # Print trainable parameters, set up MLflow, etc. only on the main process.
    # `trainer.accelerator.is_main_process` is the recommended check with HF Accelerate.
    run_id = None
    last_checkpoint = get_checkpoint(training_args)
    if trainer.accelerator.is_main_process:
        mlflow_callback.setup(training_args, trainer, model)
        node_id = trainer.accelerator.process_index
        mlflow_callback._ml_flow.system_metrics.set_system_metrics_node_id(node_id)
        run_id = mlflow_callback._ml_flow.active_run().info.run_id
        logger.info(f'Run ID: {run_id}')
        # log config file
        mlflow_callback._ml_flow.log_artifact(local_path=os.getenv('CONFIG_FILE'), artifact_path='recipe_config')
        if last_checkpoint and training_args.resume_from_checkpoint is None:    
            logger.info(f'Checkpoint detected, resuming training at {last_checkpoint}.')

    ###############
    # Training loop
    ###############
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    # Post-training metrics and model logging to MLflow only on main process
    if trainer.accelerator.is_main_process:
        if run_id is not None:
            # Collect metrics from training
            metrics = train_result.metrics
            train_samples = len(train_dataset)
            with mlflow.start_run(run_id=run_id):
                mlflow.log_param('train_samples', train_samples)
                for key, value in metrics.items():
                    mlflow.log_metric(run_id=run_id, key=key, value=value)
                # Optionally log the model
                # mlflow.transformers.log_model(
                #     run_id=run_id,
                #     transformers_model={"model": trainer.model, "tokenizer": tokenizer},
                #     artifact_path="model",
                #     registered_model_name="fine_tuned_model"
                # )
        # You can also save your final model artifacts here if desired:
        trainer.save_model(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)

    # For FSDP, ensure we save a full state dict if using PEFT
    if trainer.is_fsdp_enabled and peft_config:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type('FULL_STATE_DICT')

    trainer.model.config.use_cache = True

    if trainer.accelerator.is_main_process:
        logger.info('*** Training complete! ***')


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, SFTConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()
    set_seed(training_args.seed)
    train_function(model_args, script_args, training_args)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3.9

# This file should be deployed on the SUT to execute the various checkpoints

import argparse
import logging
import statistics
import sys
import time

PYTORCH = "pytorch"
TENSORFLOW = "tensorflow"
HUGGINGFACE = "huggingface"
DLIO = "dlio"

FRAMEWORK="framework"
SUPPORTED_FRAMEWORKS = [PYTORCH, TENSORFLOW, HUGGINGFACE, DLIO]


def parse_arguments():
    parser = argparse.ArgumentParser(description="Tool to run various checkpointing processes")
    parser.add_argument('--framework', type=str, default="dlio", help="Framework to use for checkpointing.")
    parser.add_argument('--checkpoint-location', type=str, default="/tmp", help="Path for checkpoints")
    parser.add_argument('--model', type=str, default="llama3-405b")

    parser.add_argument('--logfile-path', type=str, default="./checkpoint.log")
    return {k: v for k, v in vars(parser.parse_args()).items() if v is not None}


def validate_args(args):
    msgs = []
    if (framework := args.get(FRAMEWORK)) not in SUPPORTED_FRAMEWORKS:
        msgs.append(f'Unsupported framework "{framework}". Must be one of: {SUPPORTED_FRAMEWORKS}')

    if msgs:
        [logging.error(msg) for msg in msgs]
        sys.exit(1)


def do_dlio_checkpoint(checkpoint_location, logfile_path, num_steps=5, step_time_secs=2, model="llama3-405b", **kwargs):
    from dlio_benchmark.checkpointing.checkpointing_factory import CheckpointingFactory
    from dlio_benchmark.utils.config import ConfigArguments
    from dlio_benchmark.utils.utility import DLIOMPI
    from dlio_benchmark.common.enumerations import CheckpointMechanismType, CheckpointLocationType

    DLIOMPI.get_instance().initialize()
    comm = DLIOMPI.get_instance().comm()

    args = ConfigArguments.get_instance()
    args.logfile_path = logfile_path
    args.configure_dlio_logging(is_child=False)
    args.checkpoint_folder = checkpoint_location

    if model == "megatron":
        args.optimization_groups = [1_009_254_400, 865_075_200, 793_600]
        args.num_layers = 44
        args.layer_parameters = [129_761_280, 20_971_520]
        args.model_size = 30102
        args.checkpoint_type = CheckpointLocationType.ALL_RANKS
    elif model == "llama3-405b":
        args.num_layers = 126
        args.model_size = 16384
        args.optimization_groups = [1_009_254_400, 865_075_200, 793_600]
        args.layer_parameters = [3_214_285_714,]

    import pprint
    pprint.pprint(args)

    checkpoint_mechanism = CheckpointingFactory.get_mechanism(CheckpointMechanismType.PT_SAVE)

    for state in ["optimization_state", "model_state", "layer_state"]:
        logging.info(f'{state}: {getattr(checkpoint_mechanism, state)}')

    checkpoint_times = []
    comm.Barrier()
    for step in range(num_steps):
        logging.info(f'Starting checkpoint for step {step}...')
        start_time = time.time()
        checkpoint_mechanism.checkpoint(1, step)
        comm.Barrier()
        checkpoint_times.append(time.time() - start_time)
        logging.info(f'Wrote checkpoint in {checkpoint_times[-1]:.2f} seconds')
        time.sleep(step_time_secs)

    checkpoint_mechanism.finalize()

    logging.info(f'Average checkpoint time: {statistics.mean(checkpoint_times):.2f}')
    logging.info(f'Min & Max checkpoint times: {min(checkpoint_times):.2f}, {max(checkpoint_times):.2f}')
    logging.info(f'StDev of checkpoint times: {statistics.pstdev(checkpoint_times):.2f}')


def main(framework, **kwargs):
    if framework == DLIO:
        do_dlio_checkpoint(**kwargs)


if __name__ == "__main__":
    args = parse_arguments()
    logging.info(f'Args: {args}')

    validate_args(args)

    main(**args)

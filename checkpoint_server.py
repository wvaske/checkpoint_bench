#!/usr/bin/env python3.9

# This file should be deployed on the SUT to execute the various checkpoints

import argparse
import logging
import os
import statistics
import sys
import time
import xmlrpc.server

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
    parser.add_argument('--model', type=str, default="llama3-70b")

    parser.add_argument('--logfile-path', type=str, default="./checkpoint.log")
    return {k: v for k, v in vars(parser.parse_args()).items() if v is not None}


def validate_args(args):
    msgs = []
    if (framework := args.get(FRAMEWORK)) not in SUPPORTED_FRAMEWORKS:
        msgs.append(f'Unsupported framework "{framework}". Must be one of: {SUPPORTED_FRAMEWORKS}')

    if msgs:
        [logging.error(msg) for msg in msgs]
        sys.exit(1)


class DLIOCheckpointRPCServer:

    def __init__(self, checkpoint_location, logfile_path, model, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from dlio_benchmark.checkpointing.checkpointing_factory import CheckpointingFactory
        from dlio_benchmark.utils.config import ConfigArguments
        from dlio_benchmark.utils.utility import DLIOMPI
        from dlio_benchmark.common.enumerations import CheckpointMechanismType, CheckpointLocationType

        self.checkpoint_location = checkpoint_location
        self.logfile_path = logfile_path
        self.model = model

        self.dliompi = DLIOMPI.get_instance()
        self.dliompi.initialize()
        self.comm = self.dliompi.comm()
        self.my_rank = self.dliompi.rank()

        self.dlio_args = ConfigArguments.get_instance()
        self.dlio_args.logfile_path = logfile_path
        self.dlio_args.configure_dlio_logging(is_child=False)
        self.dlio_args.checkpoint_folder = checkpoint_location

        if model == "megatron":
            self.dlio_args.optimization_groups = [1_009_254_400, 865_075_200, 793_600]
            self.dlio_args.optimization_groups = [1_009_254, 865_075, 793]
            self.dlio_args.num_layers = 44
            # self.dlio_args.layer_parameters = [129_761_280, 20_971_520]
            # self.dlio_args.layer_parameters = [32_440_320, 5_242_880]
            self.dlio_args.layer_parameters = [16_220_160, 2_621_440]
            self.dlio_args.layer_parameters = [1_622_016, 262_144]
            self.dlio_args.model_size = 30102
            self.dlio_args.checkpoint_type = CheckpointLocationType.ALL_RANKS
            self.dlio_args.pipeline_parallelism = 2
            self.dlio_args.tensor_parallelism = 4
        elif model == "llama3-405b":
            self.dlio_args.num_layers = 80
            self.dlio_args.model_size = 16384
            self.dlio_args.optimization_groups = [1_009_254_400, 865_075_200, 793_600]
            self.dlio_args.layer_parameters = [4_358_152_160, 704_347_824]
            self.dlio_args.checkpoint_type = CheckpointLocationType.ALL_RANKS
            self.dlio_args.pipeline_parallelism = 2
            self.dlio_args.tensor_parallelism = 4
        elif model == "llama3-7b":
            self.dlio_args.num_layers = 80
            self.dlio_args.model_size = 16384
            self.dlio_args.optimization_groups = [1_009_254_4, 865_075_2, 793_6]
            self.dlio_args.layer_parameters = [4_358_152_1, 704_347_8]
            self.dlio_args.checkpoint_type = CheckpointLocationType.ALL_RANKS
            self.dlio_args.pipeline_parallelism = 2
            self.dlio_args.tensor_parallelism = 4

        self.comm.Barrier()
        self.checkpoint_mechanism = CheckpointingFactory.get_mechanism(CheckpointMechanismType.PT_SAVE)

        if self.my_rank == 0:
            for state in ["optimization_state", "model_state", "layer_state"]:
                logging.info(f'{state}: {getattr(self.checkpoint_mechanism, state)}')

        self.checkpoint_times = []
        self.comm.Barrier()

    def do_checkpoint(self, cur_step=None):
        """
        This method needs to execute a single step and return the time it took to execute the checkpoint.
        :param cur_step:
        :return:
        """
        if self.my_rank == 0:
            # Get step from input to function
            step = cur_step
        else:
            step = None

        step = self.comm.bcast(step, root=0)

        self.comm.Barrier()

        if self.my_rank == 0:
            logging.info(f'Starting checkpoint for step {step}...')

        start_time = time.time()
        self.checkpoint_mechanism.checkpoint(1, step)
        self.comm.Barrier()
        self.checkpoint_times.append(time.time() - start_time)

        if self.my_rank == 0:
            logging.info(f'Wrote checkpoint in {self.checkpoint_times[-1]:.2f} seconds')
            logging.info(f'Ending checkpoint for step {step}...')

            result_dict = dict(
                num_layers=self.dlio_args.num_layers,
                model_size=self.dlio_args.model_size,
                optimization_groups=str(self.dlio_args.optimization_groups),
                layer_parameters=str(self.dlio_args.layer_parameters),
                checkpoint_type=str(self.dlio_args.checkpoint_type),
                pipeline_parallelism=self.dlio_args.pipeline_parallelism,
                tensor_parallelism=self.dlio_args.tensor_parallelism,
                checkpoint_time=self.checkpoint_times[-1],
                step=step,
                comm_size=self.comm.size
            )

            logging.info(f'Returning result dict: {result_dict}')
            return result_dict

    def finalize(self):
        self.checkpoint_mechanism.finalize()
        if self.my_rank == 0:
            logging.info(f'METRIC - Checkpoint Times: {self.checkpoint_times}')
            logging.info(f'METRIC - Average checkpoint time: {statistics.mean(self.checkpoint_times):.2f}')
            logging.info(
                f'METRIC - Min & Max checkpoint times: {min(self.checkpoint_times):.2f}, {max(self.checkpoint_times):.2f}')
            logging.info(f'METRIC - StDev of checkpoint times: {statistics.pstdev(self.checkpoint_times):.2f}')


def main(framework, **kwargs):
    if framework == DLIO:
        checkpointer = DLIOCheckpointRPCServer(**kwargs)

        if checkpointer.my_rank != 0:
            # We just loop the do_checkpoint over and over
            while True:
                checkpointer.do_checkpoint()

        if checkpointer.my_rank == 0:
            server = xmlrpc.server.SimpleXMLRPCServer(("0.0.0.0", 8080))
            server.register_instance(checkpointer)
            server.serve_forever()


if __name__ == "__main__":
    args = parse_arguments()
    logging.info(f'Args: {args}')

    validate_args(args)

    main(**args)

#!/usr/bin/env python3.9

import argparse
import atexit
import csv
import logging
import os.path
import subprocess
import time
import xmlrpc.client


def parse_args():
    parser = argparse.ArgumentParser(description='Parse command line arguments')

    parser.add_argument("--server-ip", type=str, default="127.0.0.1",
                        help="Server IP Address for RPC Checkpoint Server")
    parser.add_argument("-p", "--port", type=str, default=8080,
                        help="Port number for RPC Checkpoint Server")
    parser.add_argument("--num-steps", type=int, default=1,
                        help="Number of steps to simulate")
    parser.add_argument("--num-passes", type=int, default=1,
                        help="Number of passes of overwriting checkpoints to simulate")
    parser.add_argument("--inter-checkpoint-sleep", type=int, default=1,
                        help="Sleep time between checkpoint operations in seconds")
    parser.add_argument("--results-dir", type=str, default="/tmp/checkpoint_bench",
                        help="Directory to store benchmark results")
    parser.add_argument("--collect-iostat", action="store_true",
                        help="Collect I/O statistics with iostat from the sysstat package")
    parser.add_argument("--iostat-interval", type=int, default=2,
                        help="Interval for collecting I/O statistics with iostat in seconds")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")

    return {k: v for k, v in vars(parser.parse_args()).items()}


def validate_args(kwargs):
    logging.info(f'Validating input arguments')
    err_msgs = []
    if kwargs.get("collect_iostat") and kwargs.get("server_ip") != "127.0.0.1":
        err_msgs.append("Collecting I/O statistics with iostat requires running the server on the same machine.")

    for msg in err_msgs:
        logging.error(msg)

    if err_msgs:
        exit(1)


class DLIOCheckpointRPCClient:

    def __init__(self, server_ip, port, num_steps, num_passes, results_dir, collect_iostat, iostat_interval=2,
                 inter_checkpoint_sleep=0, verbose=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.execution_strftime = time.strftime("%Y%m%d-%H%M%S")

        self.num_steps = num_steps
        self.server_ip = server_ip
        self.port = port
        self.collect_iostat = collect_iostat
        self.iostat_interval = iostat_interval
        self.num_passes = num_passes
        self.inter_checkpoint_sleep = inter_checkpoint_sleep
        self.verbose = verbose

        self.results_dir = os.path.join(results_dir, self.execution_strftime)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        self.results_csv_filename = os.path.join(self.results_dir, 'checkpoint_bench_results.csv')
        self.iostat_csv_filename = os.path.join(self.results_dir, f'iostat.csv')

        self.rpc_instance = None
        self.rpc_result = None
        self.iostat_process = None

        self.checkpoint_result_dicts = list()

    def setup(self):
        self.rpc_instance = xmlrpc.client.ServerProxy(f'http://{self.server_ip}:{self.port}')
        if self.collect_iostat:
            self.start_iostat_subprocess()

        atexit.register(self.teardown)

    def start_iostat_subprocess(self):
        iostat_cmd = ['iostat', '-dx', str(self.iostat_interval)]
        self.iostat_process = subprocess.Popen(iostat_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def stop_iostat_subprocess(self):
        self.iostat_process.send_signal(subprocess.signal.SIGINT)
        result = self.iostat_process.communicate()
        result_lines = result[0].splitlines()

        iostat_csv_tuples = list()
        iostat_csv_header = None
        for line in result_lines:
            line = line.decode()
            if not line:
                continue
            if line.startswith("Device") and not iostat_csv_header:
                iostat_csv_header = line.split()
            if line.startswith("Device"):
                continue

            iostat_csv_tuples.append(line.split())

        iostat_csv_tuples.insert(0, iostat_csv_header)
        logging.info(f'I/O statistics collected and saving to {self.iostat_csv_filename}')
        self.write_iostat_results_to_csv(iostat_csv_tuples)

    def write_iostat_results_to_csv(self, csv_tuples):
        with open(self.iostat_csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(csv_tuples)

    def write_result_csv(self):
        with open(self.results_csv_filename, 'w', newline='') as csvfile:
            fieldnames = self.checkpoint_result_dicts[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.checkpoint_result_dicts)

    def do_checkpoint(self, step, pass_num):
        result = self.rpc_instance.do_checkpoint(step)
        if self.verbose:
            logging.info(f'Completed checkpoint step {step} of {self.num_steps} in pass {pass_num} in {result.get("checkpoint_time"):0.1f} seconds.')
            logging.info(self.rpc_result)

        # Add some more metadata tot he result dictionary
        result['step'] = step
        result['pass_num'] = pass_num
        result['num_steps'] = self.num_steps
        result['num_passes'] = self.num_passes

        self.checkpoint_result_dicts.append(result)

    def do_checkpoint_pass(self, num_steps, pass_num):
        for step in range(1, num_steps + 1):
            logging.info(f"Starting checkpoint for step {step}/{num_steps}")
            self.do_checkpoint(step, pass_num)
            time.sleep(self.inter_checkpoint_sleep)

    def do_passes(self, num_passes=None):
        if self.collect_iostat:
            logging.info(f'Starting I/O statistics collection')
            self.start_iostat_subprocess()

        if num_passes is None:
            num_passes = self.num_passes
        for pass_num in range(1, num_passes + 1):
            logging.info(f"Starting pass {pass_num}/{args['num_passes']}")
            self.do_checkpoint_pass(self.num_steps, pass_num=pass_num)

        if self.collect_iostat:
            logging.info(f'Stopping I/O statistics collection')
            self.stop_iostat_subprocess()

    def teardown(self):
        if self.iostat_process:
            print(f'Stopping iostat...')
            self.stop_iostat_subprocess()
        logging.info(f'Writing result CSV')
        self.write_result_csv()


def main(**kwargs):
    try:
        client = DLIOCheckpointRPCClient(**args)
        client.setup()
        client.do_passes()
        client.teardown()
    except KeyboardInterrupt:
        print(f'Exiting with Ctrl+C...')
        exit(1)


if __name__ == "__main__":
    args = parse_args()
    validate_args(args)
    logging.info(f'Args: {args}')

    main(**args)



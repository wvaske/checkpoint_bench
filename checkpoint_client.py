#!/usr/bin/env python3.9

import argparse
import logging
import xmlrpc.client


def parse_args():
    parser = argparse.ArgumentParser(description='Parse command line arguments')

    parser.add_argument("--server-ip", type=str, default="127.0.0.1",
                        help="Server IP Address for RPC Checkpoint Server")
    parser.add_argument("-p", "--port", type=str, default=8080,
                        help="Port number for RPC Checkpoint Server")
    parser.add_argument("--num-steps", type=int, default=1,
                        help="Number of steps to simulate")

    return {k: v for k, v in vars(parser.parse_args()).items()}


class DLIOCheckpointRPCClient:

    def __init__(self, server_ip, port, num_steps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_steps = num_steps
        self.server_ip = server_ip
        self.port = port

        self.rpc_instance = None
        self.rpc_result = None

    def setup(self):
        self.rpc_instance = xmlrpc.client.ServerProxy(f'http://{self.server_ip}:{self.port}')

    def do_checkpoint(self, step):
        self.rpc_result = self.rpc_instance.do_checkpoint(step)
        logging.info(self.rpc_result)
        return self.rpc_result


def main(**kwargs):
    client = DLIOCheckpointRPCClient(**args)
    client.setup()

    for step in range(1, args["num_steps"] + 1):
        result = client.rpc_instance.do_checkpoint(step)
        print(f"Checkpoint result for step {step}: {result}")


if __name__ == "__main__":
    args = parse_args()
    logging.info(f'Args: {args}')

    main(**args)



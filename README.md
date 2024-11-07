## Introduction
This benchmark uses the DLIO tool from Argonne National Labs to simulate writing a checkpoint from training a large language model.

The model definition is "moderately close" to a real checkpoint. It uses torch.save() to write tensors of the proper size to the selected storage.

Each checkpoint clocks in around 440GB and is made up of 352 files. The files represent model state, 126 layers, and 

## Installation
Install DLIO from here: https://github.com/wvaske/dlio_benchmark
Be sure to do "pip install -r requirements" and "pip install dlio_benchmark"

Install an appropriate MPI library with mpirun

## Execution
Start the RPC server. 
```bash
chmod +x checkpoint_server.py

# Environment Variables may not be required and are dependent on your
# operating environment
OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
mpirun -n 8 checkpoint_server.py --framework dlio --model llama3-405b \ 
--checkpoint-location <checkpoint_location>
```

When the server shows the model_state and layer_state it is ready for executionf

The following command will execute a single checkpoint
```bash
chmod +x checkpoint_client.py

# Default IP, port and steps are localhost, 8080, and 1
./checkpoint_client.py --server-ip 127.0.0.1 --port 8080 --num-steps 55
```
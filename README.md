This benchmark uses the DLIO tool from Argonne National Labs to simulate writing a checkpoint from training a large language model.

The model definition is "moderately close" to a real checkpoint. It uses torch.save() to write tensors of the proper size to the selected storage.

## Installation
Install DLIO from here: https://github.com/wvaske/dlio_benchmark
Be sure to do "pip install -r requirements" and "pip install dlio_benchmark"

Install an appropriate MPI library with mpirun

## Execution
Start the RPC server. 
```bash
chmod +x checkpoint_server.py
OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 mpirun -n 8 checkpoint_server.py --framework dlio --model llama3-405b --checkpoint-location <checkpoint_location>
```

When the server shows the model_state and layer_state it is ready for executionf

The following command will execute a single checkpoint
```bash
./checkpoint_client.py -p 8080
```
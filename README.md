# Step-wise Federated Learning Simulation

Step-wise federated learning simulation. Source code based on [FedProx](https://github.com/litian96/FedProx). The generation of synthetic data and the basic MCLR model for machine learning is inherited from FedProx.

What's new:
- Step-wise action of client / device and server / aggregator
- Statistic Logger for received updates from clients, taking care of staleness and optimization progress (counted as number of optimizations)
- Aggregation scheme based on data portion, staleness, and progress
- Assignment of speed token to control optimization speed
- Assignment of communication token, together with different communication profiles, to control the uploading speed from client to server
- Data generation scheme specifying target standard deviation and number of classes per client

Currently zip file is encrypted and share with reviewers only. The code will be public after internal patent-related procedures are completed.


## Preparation
### Dataset generation
See [synthetic data generation](/data/synthetic_iid) for details.
### Downloading dependencies
`pip install -r requirements.txt`

## Run on synthetic data

Sample commands:
- ```
  python  -u  main.py --optimizer=fedavg --dataset=synthetic_iid \
  --learning_rate=0.02 --batch_size=8 --num_epochs=40 \
  --model=mclr --round_time=40 \
  --aggregate_with_data_size=1 \
  --data_name=floor20_std0.0_total7200_index0_NumUser30_classes_per_user2 \
  --client_profile=profile_4 --total_steps=1920 \
  --low_stoken=3 --high_stoken=4 --speed_scale=10 \
  --seed=26
  ```

- ```
  python  -u  main.py --optimizer=fedasync --dataset=synthetic_iid\
  --learning_rate=0.02 --batch_size=8 --num_epochs=40 \
  --model=mclr --eval_time_steps=40\
  --aggregate_with_data_size=1 \
  --aggregate_with_progress=1 \
  --aggregate_with_staleness=1 \
  --data_name=floor20_std0.0_total7200_index0_NumUser30_classes_per_user2 \
  --client_profile=profile_4 --total_steps=1920 \
  --low_stoken=3 --high_stoken=4 --speed_scale=10 \
  --seed=26
  ```

Script [cmd_gen.py](/cmd_gen.py) is provided to generate batch commands for control-variable experiments.

### Trainers / federated learning scheme
We provide three basic simulation scheme, all running in step-wise fashion for clients and server: _fedavg, fedasync, fedasync_buffer_. It should be specified in command `python -u main.py --optimizer=<chosen optimizer>`.

- Parameter specific to _fedavg_: `round_time`
- Parameters specific to _fedasync, fedasync_buffer_: `eval_time_steps`
- Parameter specific to _fedasync_buffer_: `async_buffer_size`

Use `total_steps` to specify the total number of discrete time steps for simulation.

### Model training parameters
- `learning_rate` default to 0.02
- `batch_size` default to 8
- `num_epochs` default to 40
- `mu` default to 0, weight of regularization when doing local optimization

### Simulated environment setting
This set of parameters determines the variability of communication link condition and local computational capacity.

**Link related**
- `client_profile`: determines the type of communication environment to be simulated. Currently available: `profile_1` to `profile_5`. See [client_profile](/clients/) for more details.


**Computation related**
- `low_stoken, high_stoken, speed_scale`: computational token are sampled in the range `[low_stoken, high_stoken) * speed_scale` to determine the number of optimizations performable by a client.
- `change_speed_interval`: determines how long the computational capacity of a client keeps steady.

### Aggregation method
`aggregate_with_data_size, aggregate_with_staleness, aggregate_with_progress`. The three parameters determine whether to use data weight, staleness weight, and progress weight in aggregation, in both asynchronous and synchronous simulations. Use binary number `0` or `1` to indicate whether to adopt the corresponding weight.

### Other parameters
- `dataset`: currently only synthetic data is used in the experiments of the paper. A version migrated to PyTorch will be shared later, supporting training convolutional neural network for image classification tasks such as MNIST and CIFAR.
- `data_name`: name of the generated dataset, different in distribution.
- `model`: currently using `mclr`, a one-layer-perceptron for classification of the synthetic data.
- `seed`: random seed for python, numpy, and tensorflow

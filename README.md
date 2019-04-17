# TensorFlow benchmarks
This repository contains various TensorFlow benchmarks. Currently, it consists of one project:

1. [scripts/tf_cnn_benchmarks](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks): The TensorFlow CNN benchmarks contain benchmarks for several convolutional neural networks.

# To run on paperspace
To run Single Worker:

1. First we have to create a project in the system and link it to https://github.com/paperspace/benchmarks
make sure to select branch: *cnn_tf_v1.13_compatible*
3. Select container:* tensorflow/tensorflow:1.13.1-gpu-py3*
4. Command: python3 scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --num_gpus=1 --model=resnet50 --variable_update=replicated --batch_size=32

5. Select Single machine with GPU. 


To run Distributed


1. Command for worker and ps: python3 scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model=resnet50 --variable_update=distributed_replicated --batch_size=32

2. Select at least 2 workers and 1 Parameter server with GPU. 


If it runs correctly it should look like this:

TensorFlow:  1.8
Model:       resnet50
Dataset:     imagenet (synthetic)
Mode:        training
SingleSess:  False
Batch size:  2048 global
             32.0 per device
Num batches: 100
Num epochs:  0.16
Devices:     ['/job:worker/replica:0/task:0/gpu:0', '/job:worker/replica:0/task:0/gpu:1', '/job:worker/replica:0/task:0/gpu:2', '/job:worker/replica:0/task:0/gpu:3', '/job:worker/replica:0/task:0/gpu:4', '/job:worker/replica:0/task:0/gpu:5', '/job:worker/replica:0/task:0/gpu:6', '/job:worker/replica:0/task:0/gpu:7']
Data format: NCHW
Layout optimizer: False
Optimizer:   momentum
Variables:   distributed_replicated
Sync:        True
==========
Generating model
W1022 16:34:12.141538 140532637939456 tf_logging.py:126] From /code/scripts/tf_cnn_benchmarks/benchmark_cnn.py:1616: Supervisor.__init__ (from tensorflow.python.training.supervisor) is deprecated and will be removed in a future version.
Instructions for updating:
Please switch to tf.train.MonitoredTrainingSession
2018-10-22 16:34:21.555656: I tensorflow/core/distributed_runtime/master_session.cc:1136] Start master session c6689e20bd4c841b with config: intra_op_parallelism_threads: 1 inter_op_parallelism_threads: 1 gpu_options { } allow_soft_placement: true
I1022 16:34:24.396881 140532637939456 tf_logging.py:116] Running local_init_op.
I1022 16:35:07.084762 140532637939456 tf_logging.py:116] Done running local_init_op.
Running warm up
Done warm up
Step    Img/sec    total_loss
1    images/sec: 324.9 +/- 0.0 (jitter = 0.0)    8.188
10    images/sec: 332.0 +/- 5.9 (jitter = 11.3)    8.166
20    images/sec: 331.1 +/- 4.1 (jitter = 14.5)    8.018
30    images/sec: 336.6 +/- 3.2 (jitter = 11.3)    7.917
40    images/sec: 339.2 +/- 2.6 (jitter = 12.6)    7.906
50    images/sec: 336.2 +/- 2.7 (jitter = 11.8)    7.929
60    images/sec: 335.8 +/- 2.4 (jitter = 11.1)    7.850
70    images/sec: 336.9 +/- 2.1 (jitter = 11.3)    7.889
80    images/sec: 337.7 +/- 1.9 (jitter = 11.3)    7.883
90    images/sec: 337.0 +/- 1.8 (jitter = 11.4)    7.896
100    images/sec: 334.7 +/- 2.0 (jitter = 10.9)    7.912
----------------------------------------------------------------
total images/sec: 2677.19
----------------------------------------------------------------




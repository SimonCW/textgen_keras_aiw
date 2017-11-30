## Introduction
I want to experiment with long short-term memory neural networks for text generation. My main motivation is to have a cool project to get comfortable with Keras while doing Andrew Ng's Deep Learning degree on coursera.
Additional learnings I hope to achieve are deepening my git skills, bash skills, and using EC2 Instances on AWS.
I'll write down my thoughts in a journey style while toying with the model.

## Prerequisites
I' ll start with this tutorial locally on my laptop. You can use the environment.yml file to create an anaconda environment that mimics my setup (tensorflow version, keras, ...). Note, however, that I created this environment on Windows. A linux system might need other dependencies. I had problems with this in the past and recommend setting up your conda environment from scratch if you use a Linux system.

## Getting Started
I'll start exploring Keras, LSTMs and text generation with this tutorial: [text-generation with lstms](https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/) and then move to AWS with this [Tutorial](https://machinelearningmastery.com/develop-evaluate-large-deep-learning-models-keras-amazon-web-services/). For help setting up your conda environment see [this tutorial](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)

The tutorial is pretty basic. I'll use it as a starting point and improve and tune the model from there.

## Reading List - Deep Dive into LSTMs
If you want to Deep Dive into LSTMs the following three are a good starting point. I recommend to read/watch the materials in that order
 - http://karpathy.github.io/2015/05/21/rnn-effectiveness/
 - http://lxmls.it.pt/2014/socher-lxmls.pdf
 - https://www.youtube.com/playlist?list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6

## Journey
### Replicating the tutorial with the Alice in Wonderland Text - 29.Nov
I commented the code more detailed than the author of the original tutorial to make sure I understand every line.
Using a CPU for this is not only non-practical but almost impossible. The writer of the Tutorial uses a Nvidia K520 GPU and trains an epoch in 300 seconds on my i7-6500U one epoch takes around 70 minutes.
Hence, I have to move earlier to the cloud than I expected. I, nevertheless, let the model run for 4 epochs. 

On an aws p2.large instance one epoch takes only about 160 seconds. These [tips](https://machinelearningmastery.com/command-line-recipes-deep-learning-amazon-web-services/) were helpful working with ec2. However, I already have a basic understanding of aws ec2.
I'm using the pre-configured Deep Learning AMI (basically a VM Image) with almost everything configured to start training on the GPU right away. The ami I use has this id ami-999844e0 in the eu-west-1 region (Ireland).
Unfortunately, something went wrong with training the model. I'll check the log file and test my code locally with a very small text input.

### Training on AWS GPU Instance - 30. Nov
My **workflow with the EC2** instance is roughly the following: 
 - Create an EBS Volume and attach it to your ec2 instance, e.g. `sudo mount /dev/xvdf /mnt/ebs`. Then make it accessible `sudo chmod -R 777 /mnt/ebs`.
 - Upload the python scripts and input files with secure copy, e.g. `scp -i ~/.ssh/*your-keypair*.pem *your-script*.py ec2-user@*public-ip*:~/`
 - Install h5py to be able to save the network weights as hdf5: sudo pip3 install h5py 
 - Run the script as background process while writing logs: nohup python3 /mnt/ebs/simple_lstm_aiw.py >/mnt/ebs/script.py.log </dev/null 2>&1 &
 - Watch the log file (also to keep your terminal busy and avoid timeout): `watch "tail script.py.log"` or watch the GPU utilization: `watch "nvidia-smi"`
 - Wait and terminate the instance after training. The model weights, tensorboard files and logs will be in the ebs file we attached in the beginning
 - For accessing the **tensorboard** created during training use `tensorboard --logdir=/path_to_your_logs`

I'll look into this example of word-level predictions next:
https://towardsdatascience.com/lstm-by-example-using-tensorflow-feb0c1968537
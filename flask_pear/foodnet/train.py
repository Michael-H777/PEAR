import data
from configs import *
import candidate

import argparse 
import warnings 
import os 
import torch 
from datetime import datetime 
import shutil 


parser = argparse.ArgumentParser(description="train file")
parser.add_argument("--preprocess", type=bool, default=False, help="process training data using specified directory")
parser.add_argument(
    "--max_lr",
    type=float,
    default=1e-2,
    help="initial learning rate, decay to end_learning_rate with cosine annihilation",
)
parser.add_argument("--max_epoch", type=int, default=100, help="maximum epoch for training")
parser.add_argument("--batch_size", type=int, default=4, help="batch size when training")
parser.add_argument(
    "--log_path", type=str, default=log_path, help="train logs location"
)
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument("--debug", type=bool, default=False, help="set to debug mode")
options = parser.parse_args()

options.DDP = True if options.gpus > 1 else False  

warnings.filterwarnings("ignore")

'''
ssh -Y -o ServerAliveInterval=6 tvy5113@submit.aci.ics.psu.edu

qsub -I -l qos=mgc_open -l nodes=1:ppn=4:gpus=4:gc_v100nvl -l walltime=72:00:00 -l pmem=gb -A eor_mri
'''


def train(rank, options):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if options.DDP:
        torch.cuda.set_device(rank)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '10000'
        
        torch.distributed.init_process_group(
            backend='nccl',
            world_size=options.gpus,
            rank=rank
        )

    # initialize data
    train_labaled_dataset = data.DataSet(filename="labeled_train.h5", percentage=(0, 0.7), augment=True)
    train_labaled_sampler = torch.utils.data.RandomSampler(
        train_labaled_dataset, replacement=True, num_samples=len(train_labaled_dataset)
    )
    train_labaled_loader = data.FastDataLoader(
        dataset=train_labaled_dataset,
        batch_size=options.batch_size,
        sampler=train_labaled_sampler,
        num_workers=4,
        drop_last=True,
        prefetch_factor=2,
    )

    train_dataset = train_labaled_dataset
    train_loader = train_labaled_loader
    
    start = 0.7
    each_range = (1-0.7) / options.gpus 
    index_range = [(start+step*each_range, start+(step+1)*each_range) for step in range(options.gpus)]
    test_dataset = data.DataSet(filename="labeled_train.h5", percentage=index_range[rank], augment=False)
    test_loader = data.FastDataLoader(
        dataset=test_dataset,
        batch_size=options.batch_size,
        num_workers=4,
        drop_last=True,
        prefetch_factor=2,
    )
    
    # initialize model
    model = candidate.single_task(name='convnetXT', ddp=options.DDP, steps=len(train_labaled_loader) * options.max_epoch)

    # initialize new folder for logs, only do this when folder not exist
    time = datetime.now().strftime("%Y_%b_%d_%p_%I_%M_%S")
    log_path = f"{options.log_path}/{model.name}_{time}"
    os.mkdir(log_path)
    [os.mkdir(f"{log_path}/{foldername}") for foldername in ["models", "test_images"]]
    shutil.copytree(os.getcwd(), f"{log_path}/code_used")
    # dump model and options

    print("-" * line_width)
    print(f"{len(train_dataset):_} images")
    print(f"{model.name} parameters: {model.compute_params()}")
    print(f"foldername: {log_path}")

    # start training
    for epoch_current, _ in enumerate(range(options.max_epoch), 1):

        start_time = datetime.now()
        model.model.train()
        print("-" * line_width)
        # do batches
        for input_data in train_loader:

            # this should not be changed, modification of loss and input should be done in model methods
            model.set_input(input_data)
            model.train()

        # exit training, do test        
        train_time = datetime.now() - start_time
        start_time = datetime.now()
        model.model.eval()
        validation = []
        for input_data in test_loader:
            model.set_input(input_data)

            acc = model.test()

            # flush image to disk
            validation.append(acc)

        # save model state_dict and check_point
        torch.save(model.model.state_dict(), f"{log_path}/models/epoch_{epoch_current}.pth")

        # use df.to_string() for convenience
        val_time = datetime.now() - start_time
        time_str = f"train: {train_time.seconds // 60 :>02}:{train_time.seconds % 60 :>02} "
        time_str += f"val: {val_time.seconds // 60 :>02}:{val_time.seconds % 60 :>02}"
        print(f'process: {rank}', time_str, f'acc: {sum(validation)/len(validation)}', sep="\n")

        # exit validation phase
    # exit training

    return None


if __name__ == "__main__":

    options.schedule = ""

    if options.preprocess:
        data.make_model_data()

    if options.debug:
        torch.autograd.set_detect_anomaly(True)
        options.max_epoch = 2
        options.epoch_samples = 40
        options.epoch_updates = 10
        options.batch_size = 4

    if options.DDP:
        torch.multiprocessing.spawn(train, args=(options, ), nprocs=options.gpus)
    else: 
        train(0, options)

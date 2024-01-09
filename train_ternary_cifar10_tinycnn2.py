import time
import copy
# PyTorch libraries
import torch

# My libraries
from config import *
from deeplearning.utils import *
from fedlearning.datasets import *
from fedlearning import *
import numpy as np

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def federated_learning(config, logger):
    """Simulate Federated Learning training process. 
    
    Args:
        config (class):          the configuration.
        logger (logging.logger): a logger for train info output.
        record (dict):           a record for train info saving.  
    """

    # initialize the dataset and dataloader for training and training 
    dataset = fetch_dataset(config)
    train_loader = fetch_dataloader(config, dataset.dst_train)
    test_loader = fetch_dataloader(config, dataset.dst_test)
    model, criterion, user_ids, user_data_mapping, start_round = init_all(config, dataset, logger)
    

    global_updater = GlobalUpdater(config)
    best_trainacc = 0.
    train_acc_list = []
    test_acc_list = []
    
    grad_size = count_parameters(model)
    print(grad_size)
    grad_c = config.clipping_bound/config.batch_size
    # mu = 2*np.sqrt(grad_size)*grad_c/np.sqrt(A*B-grad_c**2)
    # gamma = 0.56*(((B+grad_c)/B)**3*((A-grad_c)/(2*B))+((B-grad_c)/B)**3*((A+grad_c)/(2*B)))/((A/B-grad_c**2/B**2)**1.5*np.sqrt(grad_size))
    
        
    if config.optimal_config:
        A = ((1-1/config.batch_size)+np.sqrt((1/config.batch_size-1)**2+4*(4*grad_size+config.mu_target**2)/((1/config.sparse_ratio)*config.batch_size**2*config.mu_target**2)))*config.clipping_bound/2
        B = A/config.sparse_ratio
        print(A)
        print(B)
        mu_real = 2*np.sqrt(grad_size)*config.clipping_bound/np.sqrt((A-config.clipping_bound+grad_c)*B*config.batch_size**2-grad_c**2)
        gamma_real = 0.56*((1+config.clipping_bound/(B*config.batch_size))**3*((A-config.clipping_bound)/(2*B))+(1-config.clipping_bound/(B*config.batch_size))**3*((A-(config.batch_size-2)*config.clipping_bound/config.batch_size)/(2*B))+(config.clipping_bound/(B*config.batch_size))**3*(1-(A-config.clipping_bound+grad_c)/B))/(((A-config.clipping_bound+grad_c)/B-(config.clipping_bound/(B*config.batch_size))**2)**1.5*np.sqrt(grad_size))
        print("mu: {:.4}".format(mu_real))
        print("gamma: {:.6}".format(gamma_real))
        config.A = A
        config.B = B
    else:
        A = config.A
        B = config.B
        mu_real = 2*np.sqrt(grad_size)*config.clipping_bound/np.sqrt((A-config.clipping_bound+grad_c)*B*config.batch_size**2-grad_c**2)
        gamma_real = 0.56*((1+config.clipping_bound/(B*config.batch_size))**3*((A-config.clipping_bound)/(2*B))+(1-config.clipping_bound/(B*config.batch_size))**3*((A-(config.batch_size-2)*config.clipping_bound/config.batch_size)/(2*B))+(config.clipping_bound/(B*config.batch_size))**3*(1-(A-config.clipping_bound+grad_c)/B))/(((A-config.clipping_bound+grad_c)/B-(config.clipping_bound/(B*config.batch_size))**2)**1.5*np.sqrt(grad_size))
        print("mu: {:.4}".format(mu_real))
        print("gamma: {:.6}".format(gamma_real))
    
    
    if config.unbiased_indicator:
        global_learning_rate_init = copy.deepcopy(config.global_lr_init)*B
    else:
        global_learning_rate_init = copy.deepcopy(config.global_lr_init)
        
    for comm_round in range(start_round, config.rounds):
        if comm_round > 500:
            config.global_lr = global_learning_rate_init*0.125*0.5
        elif comm_round > 375:
            config.global_lr = global_learning_rate_init*0.125
        elif comm_round > 250:
            config.global_lr = global_learning_rate_init*0.25
        elif comm_round > 125:
            config.global_lr = global_learning_rate_init*0.5
        else:
            config.global_lr = copy.deepcopy(global_learning_rate_init)
        packages = {}
        user_ids_sample = np.random.choice(config.total_users, config.user_num, replace=False).tolist()
        for i, user_id in enumerate(user_ids_sample):
            updater = LocalUpdater(config, model)
            updater.init_local_dataset(dataset, user_data_mapping[user_id])
            updater.local_step(criterion)

            local_package = updater.uplink_transmit()
            packages[i] = local_package

        # Update the global model
        global_updater.global_step(model, packages)

        # Validate the model performance and log
        best_trainacc, trainacc, testacc = validate_and_log(config, model, train_loader, test_loader, criterion, comm_round, best_trainacc, logger)
        train_acc_list.append(trainacc)
        test_acc_list.append(testacc)
    
    
    log_path = './cifar10tinycnn_ternary_trainacc_log'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    trainacc_file = open(log_path + '/cifar10tinycnnalpha1trainacc_file_fed_{}_{}_{}_{}_{}_{}_{}_{}_{:.4f}_{:.4f}_{}.dat'.format(config.user_num,config.global_lr_init, config.unbiased_indicator, config.compressor, config.aggregator, config.clipping_bound, config.mu_target, config.sparse_ratio, config.A, config.B, config.mc), "w")
    for ac in train_acc_list:
        sac = str(ac)
        trainacc_file.write(sac)
        trainacc_file.write('\n')
    trainacc_file.close()
    
    log_path = './cifar10tinycnn_ternary_testacc_log'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    testacc_file = open(log_path + '/cifar10tinycnnalpha1testacc_file_fed_{}_{}_{}_{}_{}_{}_{}_{}_{:.4f}_{:.4f}_{}.dat'.format(config.user_num,config.global_lr_init, config.unbiased_indicator, config.compressor, config.aggregator, config.clipping_bound, config.mu_target, config.sparse_ratio, config.A, config.B, config.mc), "w")
    for ac in test_acc_list:
        sac = str(ac)
        testacc_file.write(sac)
        testacc_file.write('\n')
    testacc_file.close()
    
    

def main(config_file, mu, mc):
    # load the config file, logger, and initialize the output folder
    config = load_config(config_file)
    config.mu_target = mu
    config.mc = m_c
    config.sparse_ratio = 0.05
    
    output_dir = init_outputfolder(config)
    logger = init_logger(config, output_dir, config.seed)

    if config.device == "cuda":
        torch.backends.cudnn.benchmark = True

    start = time.time()
    federated_learning(config, logger)
    end = time.time()

    logger.info("{:.3} mins has elapsed".format((end-start)/60))
    logger.handlers.clear()


if __name__ == "__main__":
    config_file = "config/ternarycifar10_tinycnn2.yaml"
    mu_set = [0.1,0.5]
    for mu in mu_set:
        for m_c in range(5):
            main(config_file, mu, m_c)


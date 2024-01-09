import time

# PyTorch libraries
import torch
import copy
# My libraries
from config import *
from deeplearning.utils import *
from fedlearning.datasets import *
from fedlearning import *

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    

def federated_learning(config, logger):
    """Simulate Federated Learning training process. 
    
    Args:
        config (class):          the configuration.
        logger (logging.logger): a logger for train info output.
        record (dict):           a record for train info saving.  
    """

    # initialize the dataset and dataloader for training and testing 
    dataset = fetch_dataset(config)
    train_loader = fetch_dataloader(config, dataset.dst_train)
    test_loader = fetch_dataloader(config, dataset.dst_test)
    model, criterion, user_ids, user_data_mapping, start_round = init_all(config, dataset, logger)

    global_updater = GlobalUpdater(config)
    best_testacc = 0.
    train_acc_list = []
    test_acc_list = []
    
    d = count_parameters(model)
    std = config.noise_multiplier * config.clipping_bound
    mu = 2*config.clipping_bound/std
    print(mu)
    
    if config.unbiased_indicator:
        global_learning_rate_init = copy.deepcopy(config.global_lr_init)*config.B/config.A
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
        #print(config.global_lr)
        for i, user_id in enumerate(user_ids_sample):
            updater = LocalUpdater(config, model)
            updater.init_local_dataset(dataset, user_data_mapping[user_id])
            updater.local_step(criterion)

            local_package = updater.uplink_transmit()
            packages[i] = local_package

        # Update the global model
        global_updater.global_step(model, packages)

        # Validate the model performance and log
        best_testacc, trainacc, testacc = validate_and_log(config, model, train_loader, test_loader, criterion, comm_round, best_testacc, logger)
        train_acc_list.append(trainacc)
        test_acc_list.append(testacc)

    
    
    log_path = './cifar10tinycnn_dpsgd_trainacc_log'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    trainacc_file = open(log_path + '/cifar10tinycnnalpha1trainacc_file_fed_{}_{}_{}_{}_{}_{}_{:.4f}_{}_{}_{}.dat'.format(config.user_num, config.global_lr_init, config.unbiased_indicator, config.compressor, config.aggregator, config.clipping_bound, mu, config.A, config.B, config.mc), "w")
    for ac in train_acc_list:
        sac = str(ac)
        trainacc_file.write(sac)
        trainacc_file.write('\n')
    trainacc_file.close()
    
    log_path = './cifar10tinycnn_dpsgd_testacc_log'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    testacc_file = open(log_path + '/cifar10tinycnnalpha1testacc_file_fed_{}_{}_{}_{}_{}_{}_{:.4f}_{}_{}_{}.dat'.format(config.user_num, config.global_lr_init, config.unbiased_indicator, config.compressor, config.aggregator, config.clipping_bound, mu, config.A, config.B, config.mc), "w")
    for ac in test_acc_list:
        sac = str(ac)
        testacc_file.write(sac)
        testacc_file.write('\n')
    testacc_file.close()
    

def main(config_file, mu, mc):
    # load the config file, logger, and initialize the output folder
    config = load_config(config_file)
    config.noise_multiplier = 2/mu
    config.mc = m_c
    config.A = 0.005
    config.B = 0.1
    
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
    config_file = "config/fedavgcifar10_tinycnn.yaml"
    mu_set = [0.1,0.5]
    for mu in mu_set:
        for m_c in range(5):
            main(config_file, mu, m_c)


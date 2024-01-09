from opacus import PrivacyEngine

from fedlearning.dp.my_dp_optimizer import MyDPOptimizer

def prepare_dp(config, model, optimizer, data_loader):
    privacy_engine = PrivacyEngine()

    # model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
    #     module=model,
    #     optimizer=optimizer,
    #     data_loader=data_loader,
    #     epochs=config.local_epochs,
    #     target_epsilon=config.epsilon,
    #     target_delta=config.delta,
    #     max_grad_norm=1
    # )

    model, optimizer, data_loader = privacy_engine.make_private(
            module = model,
            optimizer = optimizer,
            data_loader = data_loader,
            noise_multiplier = config.noise_multiplier,
            max_grad_norm = config.clipping_bound,
            poisson_sampling=False)

    return model, optimizer, data_loader

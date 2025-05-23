import os
import torch
import time
import numpy as np
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.functional import F
import logging

log = logging.getLogger('main')

from aime_nob.data import SequenceDataset, MultiDataset, get_sample_loader
from aime_nob.utils import *
from aime_nob.models.base import MLP
from aime_nob.models.ssm import ssm_classes
from aime_nob.models.policy import TanhGaussianPolicy
from aime_nob.logger import get_default_logger
from aime_nob.env import env_classes, SaveTrajectories
from aime_nob.actor import PolicyActor, RandomActor, GuassianNoiseActorWrapper
from aime_nob.models.value import VNetDict
from aime_nob.runtimes import runtime_classes


def compute_gradient_penalty(discriminator, expert_data, policy_data, penalty_scale=10):
    alpha = torch.rand(*expert_data.shape[:-1], 1).to(expert_data)

    mixup_data = alpha * expert_data + (1 - alpha) * policy_data
    mixup_data.requires_grad = True

    disc = discriminator(mixup_data)
    ones = torch.ones_like(disc)
    grad = torch.autograd.grad(outputs=disc, inputs=mixup_data, grad_outputs=ones, create_graph=True, retain_graph=True, only_inputs=True)[0]

    grad_pen = penalty_scale * (grad.norm(2, dim=-1) - 1).pow(2).mean()
    return grad_pen


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name='aime-nob')
def main(config: DictConfig):
    runtime = runtime_classes[config['runtime']['name']](config)

    setup_seed(config['seed'])
    fp16_dtype = torch.bfloat16 if config['fp16_dtype'] == 'bfloat16' else torch.float16

    log.info('Using the following config:\n' + OmegaConf.to_yaml(config))

    log_name = config['log_name']
    output_folder = os.path.join(OUTPUT_PATH, log_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    log.info(f'Log will be stored at {output_folder}')

    OmegaConf.save(config, os.path.join(output_folder, 'config.yaml'))
    demonstration_dataset_folder = os.path.join(DATA_PATH, config['demonstration_dataset_name'])
    experience_folder = os.path.join(output_folder, 'train_trajectories')
    eval_folder = os.path.join(output_folder, 'eval_trajectories')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info(f'using device {device}')

    log.info('Creating environments ...')
    env_config = dict(config['env'])
    env_config['seed'] = config['seed']
    env_class_name = env_config.pop('class')
    env = env_classes[env_class_name](**env_config)
    render = env_config['render'] or need_render(config['environment_setup'])
    env.enable_render(render)
    sensor_layout = env_config['sensors']
    env.action_space.seed(config['seed'])
    if env.multi_instancable:
        env_config = dict(env_config)
        env_config['seed'] *= 2
        test_env = env_classes[env_class_name](**env_config)
        test_env.enable_render(True)
        test_env = SaveTrajectories(test_env, eval_folder)
    else:
        # NOTE: This is creating a bug that the evalutation trajectories will also go into the reply buffer.
        test_env = SaveTrajectories(env, eval_folder)

    sensor_layout = env_config['sensors']
    world_model_config = parse_world_model_config(config, sensor_layout, env.observation_space, predict_terminal=config['use_terminal'], predict_reward=True, use_probe=config['use_probe'])
    # replace the reward head with ot reward head
    for i, c in enumerate(world_model_config['output_config']):
        if c[0] == 'reward':
            c = list(c)
            c[0] = 'ail_reward'
            world_model_config['output_config'][i] = tuple(c)
    selected_keys = get_seleted_keys_from_world_model_config(world_model_config)
    selected_keys = selected_keys + ['reward']
    world_model_name = world_model_config.pop('name')
    image_sensors, used_image_sensors = get_image_sensors(world_model_config, sensor_layout)
    model = ssm_classes[world_model_name](**world_model_config)
    if config['pretrained_model_name'] is not None:
        pretrained_paramters = torch.load(os.path.join(MODEL_PATH, config['pretrained_model_name'], 'model.pt'), map_location='cpu')
        model.load_state_dict(pretrained_paramters, strict=False)
        if config['freeze_pretrained_parameters']:
            for name, parameter in model.named_parameters():
                if name in pretrained_paramters.keys():
                    parameter.requires_grad_(False)
    log.info('Model Information:\n' + str(torchinfo.summary(model, verbose=0, row_settings=['var_names', 'depth'])))
    model = model.to(device)

    dataset = SequenceDataset(experience_folder, config['horizon'], overlap=True, selected_keys=selected_keys, **config['data']['dataset'])
    eval_dataset = SequenceDataset(eval_folder, config['horizon'], overlap=False)
    expert_dataset = SequenceDataset(
        demonstration_dataset_folder, config['horizon'], overlap=True, max_capacity=config['num_expert_trajectories'], selected_keys=selected_keys, **config['data']['dataset']
    )
    log.info(f'Training on {len(expert_dataset.trajectories)} expert trajectories!')

    env = SaveTrajectories(env, experience_folder)

    # collect initial dataset
    for _ in range(config['prefill']):
        actor = RandomActor(env.action_space)
        interact_with_environment(env, actor, [])
    dataset.update()

    # add old embodiment dataset if available
    if config['embodiment_dataset_name'] is not None:
        log.info('Loading embodiment dataset ...')
        embodiment_dataset_folder = os.path.join(DATA_PATH, config['embodiment_dataset_name'])
        embodiment_dataset = make_static_dataset(embodiment_dataset_folder, config['horizon'], overlap=True, selected_keys=selected_keys, **config['data']['dataset'])
        if config['num_embodiment_trajectories'] is not None:
            embodiment_dataset.keep(config['num_embodiment_trajectories'], config['embodiment_dataset_sampling_mode'])
        if config['embodiment_dataset_reg_mode'] == 'append':
            dataset = MultiDataset([embodiment_dataset, dataset])

    policy_config = config['policy']
    policy = TanhGaussianPolicy(model.state_feature_dim, world_model_config['action_dim'], **policy_config)
    if config['pretrained_model_name'] is not None and config['load_policy']:
        policy.load_state_dict(torch.load(os.path.join(MODEL_PATH, config['pretrained_model_name'], 'policy.pt'), map_location='cpu'), strict=False)
    policy = policy.to(device)

    model_optim = model.get_optimizor(dict(world_model_config['optimizor']))
    model_scaler = torch.cuda.amp.GradScaler(enabled=config['use_fp16'])
    policy_optim = torch.optim.Adam(policy.parameters(), lr=config['policy_lr'])
    policy_scaler = torch.cuda.amp.GradScaler(enabled=config['use_fp16'])

    vnet_config = config['vnet']
    reward_keys = ['ail_reward']
    if not model.intrinsic_reward_disable:
        reward_keys.append('intrinsic_reward')
    vnet = VNetDict(model.state_feature_dim, reward_keys, **vnet_config)
    if config['pretrained_model_name'] is not None and config['load_vnet']:
        vnet.load_state_dict(torch.load(os.path.join(MODEL_PATH, config['pretrained_model_name'], 'vnet.pt'), map_location='cpu'), strict=False)
    vnet = vnet.to(device)
    vnet_optim = torch.optim.Adam(vnet.parameters(), lr=config['vnet_lr'])
    vnet_scaler = torch.cuda.amp.GradScaler(enabled=config['use_fp16'])

    discriminator = MLP(model.emb_dim, 1, config['ail_config']['hidden_size'], config['ail_config']['hidden_layers'])
    discriminator = discriminator.to(device)
    discriminator_optim = torch.optim.Adam(discriminator.parameters(), lr=config['ail_config']['discriminator_lr'])
    discriminator_scaler = torch.cuda.amp.GradScaler(enabled=config['use_fp16'])

    logger = get_default_logger(output_folder)
    additional_logger = runtime.get_additional_logger(output_folder)
    if additional_logger is not None:
        logger.append(additional_logger)

    def train_model(iterations):
        train_metric_tracker = AverageMeter()
        training_start_time = time.time()

        def train_model_on_data(embodiment_data):
            embodiment_data = embodiment_data.to(device)
            with torch.autocast(device_type=device, dtype=fp16_dtype, enabled=config['use_fp16']):
                _, _, loss, _metrics = model(embodiment_data, embodiment_data['pre_action'])

            model_optim.zero_grad(set_to_none=True)
            model_scaler.scale(loss).backward()
            model_scaler.unscale_(model_optim)
            grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), config['grad_clip'])
            model_scaler.step(model_optim)
            model_scaler.update()
            _metrics['model_grad_norm'] = grad_norm.item()
            _metrics = {'model/' + k: v for k, v in _metrics.items()}
            train_metric_tracker.add(_metrics)

        if config['embodiment_dataset_reg_mode'] == 'append':
            embodiment_loader = get_sample_loader(dataset, config['batch_size'], iterations, **config['data']['dataloader'])
            for embodiment_data in tqdm(iter(embodiment_loader), disable=runtime.disable_tqdm):
                train_model_on_data(embodiment_data)
        elif config['embodiment_dataset_reg_mode'] == 'ratio':
            ratio = config['embodiment_dataset_reg_sampling_ratio']
            reg_data_batch_size = int(config['batch_size'] * ratio)
            online_data_batch_size = config['batch_size'] - reg_data_batch_size
            reg_embodiment_loader = get_sample_loader(embodiment_dataset, reg_data_batch_size, iterations, **config['data']['dataloader'])
            online_embodiment_loader = get_sample_loader(dataset, online_data_batch_size, iterations, **config['data']['dataloader'])
            for reg_embodiment_data, online_embodiment_data in tqdm(zip(iter(reg_embodiment_loader), iter(online_embodiment_loader)), disable=runtime.disable_tqdm, total=config['batch_per_epoch']):
                embodiment_data = ArrayDict.cat([reg_embodiment_data, online_embodiment_data], dim=1)
                train_model_on_data(embodiment_data)
        else:
            raise NotImplementedError

        log.info(f'Model training last for {time.time() - training_start_time:.3f} s')
        return train_metric_tracker.get()

    if config['model_pretraining_iterations'] > 0:
        log.info(f'pretrain the model for {config["model_pretraining_iterations"]} iterations.')
        train_model(config['model_pretraining_iterations'])

    if config['policy_pretraining_iterations'] > 0:
        log.info(f'pretrain the policy for {config["policy_pretraining_iterations"]} iterations.')
        loader = get_sample_loader(expert_dataset, config['batch_size'], config['policy_pretraining_iterations'], **config['data']['dataloader'])
        for data in tqdm(iter(loader), disable=runtime.disable_tqdm):
            data = data.to(device)

            with torch.autocast(device_type=device, dtype=fp16_dtype, enabled=config['use_fp16']):
                outputs, _, action_seq, loss, _metrics = model.filter_with_policy(data, policy, None, kl_only=config['kl_only'])
                policy_entropy_loss = -config['policy_entropy_scale'] * torch.mean(outputs['action_entropy'].sum(dim=-1))
                loss = loss + policy_entropy_loss

            policy_optim.zero_grad(set_to_none=True)
            policy_scaler.scale(loss).backward()
            policy_scaler.unscale_(policy_optim)
            grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(policy.parameters(), config['grad_clip'])
            policy_scaler.step(policy_optim)
            policy_scaler.update()

    for e in range(config['prefill'], config['epoch']):
        log.info(f'Starting epcoh {e}')

        metrics = {}

        if config['policy_reset_period'] is not None and (e + config['prefill']) % config['policy_reset_period'] == 0:
            log.info('Resetting Policy ...')
            policy = TanhGaussianPolicy(model.state_feature_dim, world_model_config['action_dim'], **policy_config)
            policy = policy.to(device)
            policy_optim = torch.optim.Adam(policy.parameters(), lr=config['policy_lr'])
            policy_scaler = torch.cuda.amp.GradScaler(enabled=config['use_fp16'])

        if not config['offline']:
            log.info('Collecting new data ...')
            with torch.no_grad():
                actor = PolicyActor(model, policy, eval=False)
                actor = GuassianNoiseActorWrapper(actor, config['epsilon'], env.action_space)
                result = interact_with_environment(env, actor, image_sensors)
                metrics.update({f'train_{k}': v for k, v in result.items()})

            dataset.update()

        log.info('Training Model and Policy ...')
        train_metric_tracker = AverageMeter()
        training_start_time = time.time()

        def train_model_and_policy(embodiment_data, demonstration_data):
            metrics = {}
            embodiment_data = embodiment_data.to(device)
            demonstration_data = demonstration_data.to(device)

            if 'reward' in demonstration_data.keys():
                demonstration_data['reward_mask'] = torch.zeros_like(demonstration_data['reward'])

            # train the discriminator first
            with torch.autocast(device_type=device, dtype=fp16_dtype, enabled=config['use_fp16']):
                with torch.no_grad():
                    demonstration_emb = model.get_emb(demonstration_data)

                embodiment_emb = model.get_emb(embodiment_data)

                demonstration_discriminator_score = discriminator(demonstration_emb)
                embodiment_discriminator_score = discriminator(embodiment_emb.detach())

                discriminator_loss = F.binary_cross_entropy_with_logits(demonstration_discriminator_score, torch.ones_like(demonstration_discriminator_score)) + F.binary_cross_entropy_with_logits(
                    embodiment_discriminator_score, torch.zeros_like(embodiment_discriminator_score)
                )

                grad_penalty = compute_gradient_penalty(discriminator, demonstration_emb.detach(), embodiment_emb.detach(), config['ail_config']['penalty_scale'])

            discriminator_optim.zero_grad(set_to_none=True)
            discriminator_scaler.scale(discriminator_loss + grad_penalty).backward()
            discriminator_scaler.unscale_(discriminator_optim)
            grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(discriminator.parameters(), config['grad_clip'])
            discriminator_scaler.step(discriminator_optim)
            discriminator_scaler.update()
            metrics['discriminator/embodiment_score'] = torch.sigmoid(embodiment_discriminator_score).mean().item()
            metrics['discriminator/demonstration_score'] = torch.sigmoid(demonstration_discriminator_score).mean().item()
            metrics['discriminator/grad_norm'] = grad_norm.item()
            metrics['discriminator/loss'] = discriminator_loss.item()
            metrics['discriminator/grad_penalty'] = grad_penalty.item()

            with torch.no_grad():
                embodiment_discriminator_score = discriminator(embodiment_emb.detach())
                embodiment_discriminator_score = torch.sigmoid(embodiment_discriminator_score)
                embodiment_data['ail_reward'] = embodiment_discriminator_score.log() - (1 - embodiment_discriminator_score).log()

            # then to train model
            with torch.autocast(device_type=device, dtype=fp16_dtype, enabled=config['use_fp16']):
                _, state_seq, model_loss, model_metrics = model(embodiment_data, embodiment_data['pre_action'], emb_seq=embodiment_emb)

            model_optim.zero_grad(set_to_none=True)
            model_scaler.scale(model_loss).backward()
            model_scaler.unscale_(model_optim)
            grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), config['grad_clip'])
            model_scaler.step(model_optim)
            model_scaler.update()
            model_metrics['model_grad_norm'] = grad_norm.item()
            model_metrics = {'model/' + k: v for k, v in model_metrics.items()}
            metrics.update(model_metrics)

            # then to train the policy
            states = model.flatten_states(state_seq)
            states.vmap_(lambda v: v.detach())

            with torch.autocast(device_type=device, dtype=fp16_dtype, enabled=config['use_fp16']):
                # compute aime loss
                demonstration_pre_action = demonstration_data.pop('pre_action')
                outputs, _, action_seq, aime_loss, policy_metrics = model.filter_with_policy(demonstration_data, policy, None, kl_only=config['kl_only'])
                demonstration_data['pre_action'] = demonstration_pre_action
                # you should not be able to compute this metric in the real setting, we compute here only for analysis
                policy_metrics['action_mse'] = model.metric_func(demonstration_pre_action, action_seq).item()
                policy_entropy_loss = -config['policy_entropy_scale'] * torch.mean(outputs['action_entropy'].sum(dim=-1))
                policy_loss = config['aime_gradient_loss_scale'] * (aime_loss + policy_entropy_loss)
                policy_metrics['aime_action_entropy_loss'] = policy_entropy_loss.item()

                # compute value gradient loss
                state_seq, _, outputs = model.rollout_with_policy(states, policy, config['imagine_horizon'], names=[*reward_keys, 'is_terminal'], state_detach=True, action_sample=True)

                state_features = torch.stack([model.get_state_feature(state) for state in state_seq])
                target_value_dict = vnet.compute_target(state_features)

                value_gradient_loss = 0
                target_return_dict = {}
                discount = config['gamma'] * (1 - outputs['is_terminal'])
                cum_discount = torch.cumprod(torch.cat([torch.ones_like(discount[:1]), discount[:-1]], dim=0), dim=0)

                for reward_key in reward_keys:
                    reward = outputs[reward_key]
                    value = target_value_dict[reward_key]

                    target_return_dict[reward_key] = lambda_return(reward[:-1], value[:-1], discount[:-1], value[-1], config['lambda'])
                    if env.action_type == 'continuous':
                        _value_gradient_loss = -torch.mean(cum_discount[:-2] * target_return_dict[reward_key][1:])
                    elif env.action_type == 'discrete':
                        advantage = (target_return_dict[reward_key][1:] - value[:-2]).detach()
                        _value_gradient_loss = -torch.mean(cum_discount[:-2] * outputs['action_logp'][:-1] * advantage)
                    value_gradient_loss = value_gradient_loss + _value_gradient_loss
                    policy_metrics[f'value_gradient_policy_loss_{reward_key}'] = _value_gradient_loss.item()

                    policy_metrics[f'value_gradient_policy_loss'] = value_gradient_loss.item()
                    policy_entropy_loss = -config['policy_entropy_scale'] * torch.mean(outputs['action_entropy'].sum(dim=-1))
                    policy_metrics['value_gradient_policy_entropy_loss'] = policy_entropy_loss.item()
                    value_gradient_loss = value_gradient_loss + policy_entropy_loss
                    policy_loss = policy_loss + value_gradient_loss * config['value_gradient_loss_scale']

            policy_optim.zero_grad(set_to_none=True)
            policy_scaler.scale(policy_loss).backward()
            policy_scaler.unscale_(policy_optim)
            grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(policy.parameters(), config['grad_clip'])
            policy_scaler.step(policy_optim)
            policy_scaler.update()
            policy_metrics['policy_grad_norm'] = grad_norm.item()
            policy_metrics = {'policy/' + k: v for k, v in policy_metrics.items()}
            metrics.update(policy_metrics)

            # finally train the value function
            with torch.autocast(device_type=device, dtype=fp16_dtype, enabled=config['use_fp16']):
                value_dict = vnet(state_features[:-1].detach())
                value_loss = 0
                for reward_key in reward_keys:
                    _value_loss = 0.5 * torch.mean((target_return_dict[reward_key].detach() - value_dict[reward_key]) ** 2 * cum_discount[:-1].detach())
                    value_loss = value_loss + _value_loss
                    metrics[f'value/value_{reward_key}'] = value_dict[reward_key].mean().item()
                    metrics[f'value/value_loss_{reward_key}'] = _value_loss.item()
                metrics['value/value_loss'] = value_loss.item()

            vnet_optim.zero_grad(set_to_none=True)
            vnet_scaler.scale(value_loss).backward()
            vnet_scaler.unscale_(vnet_optim)
            grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(vnet.parameters(), config['grad_clip'])
            vnet_scaler.step(vnet_optim)
            vnet_scaler.update()
            metrics['value/vnet_grad_norm'] = grad_norm.item()
            vnet.update_target()

            train_metric_tracker.add(metrics)

        demonstration_loader = get_sample_loader(expert_dataset, config['batch_size'], config['batch_per_epoch'], **config['data']['dataloader'])

        if config['embodiment_dataset_reg_mode'] == 'append' or config['embodiment_dataset_name'] is None:
            embodiment_loader = get_sample_loader(dataset, config['batch_size'], config['batch_per_epoch'], **config['data']['dataloader'])
            for embodiment_data, demonstration_data in tqdm(zip(iter(embodiment_loader), iter(demonstration_loader)), disable=runtime.disable_tqdm, total=config['batch_per_epoch']):
                train_model_and_policy(embodiment_data, demonstration_data)
        elif config['embodiment_dataset_reg_mode'] == 'ratio':
            ratio = config['embodiment_dataset_reg_sampling_ratio']
            reg_data_batch_size = int(config['batch_size'] * ratio)
            online_data_batch_size = config['batch_size'] - reg_data_batch_size
            reg_embodiment_loader = get_sample_loader(embodiment_dataset, reg_data_batch_size, config['batch_per_epoch'], **config['data']['dataloader'])
            online_embodiment_loader = get_sample_loader(dataset, online_data_batch_size, config['batch_per_epoch'], **config['data']['dataloader'])
            for reg_embodiment_data, online_embodiment_data, demonstration_data in tqdm(
                zip(iter(reg_embodiment_loader), iter(online_embodiment_loader), iter(demonstration_loader)), disable=runtime.disable_tqdm, total=config['batch_per_epoch']
            ):
                embodiment_data = ArrayDict.cat([reg_embodiment_data, online_embodiment_data], dim=1)
                train_model_and_policy(embodiment_data, demonstration_data)
        else:
            raise NotImplementedError

        metrics.update(train_metric_tracker.get())
        log.info(f'Model and policy training last for {time.time() - training_start_time:.3f} s')

        if e % config['test_period'] == 0:
            log.info('Evaluating the model ...')
            with torch.no_grad():
                actor = PolicyActor(model, policy, eval=True)
                metrics.update(eval_actor_on_env(test_env, actor, image_sensors, config['num_test_trajectories']))
            eval_dataset.update()
            for image_key in image_sensors:
                metrics[f'eval_video_{image_key}'] = eval_dataset.get_trajectory(-1)[image_key].permute(0, 2, 3, 1).contiguous() * 255

            if len(used_image_sensors) > 0 or (test_env.set_state_from_obs_support and test_env.check_set_state_from_obs_requirements(model.output_names)):
                log.info('Generating prediction videos ...')
                metrics.update(generate_prediction_videos(model, demonstration_data, test_env, image_sensors, used_image_sensors, 10, 6))

            log.info('Saving the models ...')
            torch.save(model.state_dict(), os.path.join(output_folder, 'model.pt'))
            torch.save(policy.state_dict(), os.path.join(output_folder, 'policy.pt'))
            torch.save(vnet.state_dict(), os.path.join(output_folder, 'vnet.pt'))
            torch.save(discriminator.state_dict(), os.path.join(output_folder, 'discriminator.pt'))

        logger(metrics, e)

        runtime.upload(e, output_folder)

    log.info('Evaluating the final model ...')
    metrics = {}
    with torch.no_grad():
        actor = PolicyActor(model, policy, eval=True)
        metrics.update(eval_actor_on_env(test_env, actor, image_sensors, config['final_num_test_trajectories']))
    eval_dataset.update()
    for image_key in image_sensors:
        metrics[f'eval_video_{image_key}'] = eval_dataset.get_trajectory(-1)[image_key].permute(0, 2, 3, 1).contiguous() * 255

    if len(used_image_sensors) > 0 or (test_env.set_state_from_obs_support and test_env.check_set_state_from_obs_requirements(model.output_names)):
        log.info('Generating prediction videos ...')
        metrics.update(generate_prediction_videos(model, demonstration_data, test_env, image_sensors, used_image_sensors, 10, 6))

    log.info('Saving the final models ...')
    torch.save(model.state_dict(), os.path.join(output_folder, 'model.pt'))
    torch.save(policy.state_dict(), os.path.join(output_folder, 'policy.pt'))
    torch.save(vnet.state_dict(), os.path.join(output_folder, 'vnet.pt'))
    torch.save(discriminator.state_dict(), os.path.join(output_folder, 'discriminator.pt'))

    logger(metrics, e + 1)

    runtime.finish(output_folder)


if __name__ == '__main__':
    main()

import argparse
from collections import defaultdict
from copy import deepcopy
from distutils.util import strtobool
import json
import os
import pickle as pkl
import random

import numpy as np
import torch
import torch.backends.cudnn
from torchvision.transforms import transforms
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:  # type: ignore[override]
        """No-op fallback when tensorboard is unavailable in the runtime."""

        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def add_text(self, *args, **kwargs):
            pass

        def close(self):
            pass

from competitors.dann.dann import train_dann
from competitors.jumbot.jumbot import train_jumbot
from competitors.mmd.train_mmd import train_mmd
from competitors.alda.train_alda import train_alda
from dataset import PixelSetData, create_evaluation_loaders, create_train_loader
from evaluation import evaluation, validation
from ideas.source_feature_reshaper import (
    build_source_feature_reshaper,
    forward_with_optional_source_reshaper,
)
from ideas.train_source_phase_compactness import train_supervised_source_phase_compactness
from models.stclassifier import PseLTae, PseTae, PseTempCNN, PseGru
from timematch import train_timematch
from transforms import (
    Normalize,
    RandomSamplePixels,
    RandomSampleTimeSteps,
    ToTensor,
    RandomTemporalShift,
    Identity,
)
from utils import label_utils
from utils.focal_loss import FocalLoss
from utils.metrics import overall_classification_report
from utils.train_utils import AverageMeter, bool_flag, to_cuda



def main(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)

    # Select classes that appear at least 200 times source
    source_classes = label_utils.get_classes(cfg.source.split('/')[0], combine_spring_and_winter=cfg.combine_spring_and_winter)
    if config.closed_set:
        source_classes = [cls for cls in source_classes if cls != 'unknown']
    source_data = PixelSetData(cfg.data_root, cfg.source, source_classes, closed_set=config.closed_set)
    labels, counts = np.unique(source_data.get_labels(), return_counts=True)
    source_classes = [source_classes[i] for i in labels[counts >= 200]]
    print('Using classes:', source_classes)
    cfg.classes = source_classes
    cfg.num_classes = len(source_classes)

    # Randomly assign parcels to train/val/test
    indices = {
        config.source: len(source_data),
        config.target: len(PixelSetData(config.data_root, config.target, source_classes, closed_set=config.closed_set))
    }
    folds = create_train_val_test_folds([config.source, config.target], config.num_folds, indices, config.val_ratio, config.test_ratio)

    if config.overall:
        overall_performance(config)
        return

    for fold_num, splits in enumerate(folds):
        print(f'Starting fold {fold_num}...')

        config.fold_dir = os.path.join(config.output_dir, f'fold_{fold_num}')
        config.fold_num = fold_num

        sample_pixels_val = config.sample_pixels_val or (config.eval and config.temporal_shift)
        val_loader, test_loader = create_evaluation_loaders(config.target, splits, config, sample_pixels_val)

        if config.model == 'pseltae':
            model = PseLTae(input_dim=config.input_dim, num_classes=config.num_classes, with_extra=config.with_extra)
        elif config.model == 'psetae':
            model = PseTae(input_dim=config.input_dim, num_classes=config.num_classes, with_extra=config.with_extra)
        elif config.model == 'psetcnn':
            model = PseTempCNN(input_dim=config.input_dim, num_classes=config.num_classes, with_extra=config.with_extra)
        elif config.model == 'psegru':
            model = PseGru(input_dim=config.input_dim, num_classes=config.num_classes, with_extra=config.with_extra)
        else:
            raise NotImplementedError()
        
        model.to(config.device)

        best_model_path = os.path.join(config.fold_dir, 'model.pt')

        if not config.eval:
            print(model)
            print('Number of trainable parameters:', get_num_trainable_params(model))

            # if os.path.isfile(best_model_path):
            #     answer = input(f'Model already exists at {best_model_path}! Override y/[n]? ')
            #     override = strtobool(answer) if len(answer) > 0 else False
            #     if not override:
            #         print('Skipping fold', fold_num)
            #         continue

            writer = SummaryWriter(log_dir=f'{config.tensorboard_log_dir}_fold{fold_num}', purge_step=0)
            if config.method == 'timematch':
                train_timematch(model, config, writer, val_loader, device, best_model_path, fold_num, splits)
            elif config.method == 'dann':
                train_dann(model, config, writer, val_loader, device, best_model_path, fold_num, splits)
            elif config.method == 'mmd':
                train_mmd(model, config, writer, val_loader, device, best_model_path, fold_num, splits)
            elif config.method == 'jumbot':
                train_jumbot(model, config, writer, val_loader, device, best_model_path, fold_num, splits)
            elif config.method == 'alda':
                train_alda(model, config, writer, val_loader, device, best_model_path, fold_num, splits)
            elif config.method == 'sourcephasecompact':
                train_supervised_source_phase_compactness(model, config, writer, splits, val_loader, device, best_model_path)
            else:
                train_supervised(model, config, writer, splits, val_loader, device, best_model_path)

        if config.skip_final_test:
            print('Skipping final labeled test/eval because --skip_final_test=True')
            continue

        print('Restoring best model weights for testing...')

        checkpoint = torch.load(best_model_path, weights_only=False)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        source_feature_reshaper = maybe_build_source_feature_reshaper(model, config)
        if source_feature_reshaper is not None:
            source_feature_reshaper.to(device)
            if 'source_feature_reshaper_state_dict' in checkpoint:
                source_feature_reshaper.load_state_dict(checkpoint['source_feature_reshaper_state_dict'])

        test_metrics = evaluation(
            model,
            test_loader,
            device,
            config.classes,
            mode='test',
            source_feature_reshaper=source_feature_reshaper,
            apply_source_feature_reshaper=False,
        )

        print(f"Test result for {config.experiment_name}: accuracy={test_metrics['accuracy']:.4f}, f1={test_metrics['macro_f1']:.4f}")
        print(test_metrics['classification_report'])

        save_results(test_metrics, config)

    if config.skip_final_test:
        print('Skipping overall_performance aggregation because --skip_final_test=True')
        return

    overall_performance(config)


def get_num_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def maybe_build_source_feature_reshaper(model, config):
    return build_source_feature_reshaper(
        kind=getattr(config, "source_feature_reshaper", "none"),
        feature_dim=model.spatial_encoder.output_dim,
        strength=getattr(config, "source_feature_reshaper_strength", 0.10),
        kernel_size=getattr(config, "source_feature_reshaper_kernel_size", 3),
    )

def get_dataset_size(data_root, dataset):
    dir = os.path.join(data_root, dataset)
    return len([name for name in os.listdir(os.path.join(dir, 'data')) if name.endswith('.zarr')])

def train_supervised(model, config, writer, splits, val_loader, device, best_model_path):
    model.to(device)

    best_f1 = 0
    dataset_name = config.source
    if config.train_on_target:
        dataset_name = config.target

    source_feature_reshaper = maybe_build_source_feature_reshaper(model, config)
    params = list(model.parameters())
    if source_feature_reshaper is not None:
        source_feature_reshaper.to(device)
        params += list(source_feature_reshaper.parameters())
    optimizer = torch.optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)

    train_transform = transforms.Compose([
        RandomSamplePixels(config.num_pixels),
        RandomSampleTimeSteps(config.seq_length),
        RandomTemporalShift(max_shift=config.max_shift_aug, p=config.shift_aug_p) if config.with_shift_aug else Identity(),
        Normalize(),
        ToTensor(),
    ])

    dataset = PixelSetData(
        config.data_root,
        dataset_name,
        config.classes,
        train_transform,
        splits[dataset_name]['train'],
        closed_set=config.closed_set,
    )
    data_loader = create_train_loader(dataset, config.batch_size, config.num_workers)
    print(f'training dataset: {dataset_name}, n={len(dataset)}, batches={len(data_loader)}')

    criterion = FocalLoss(gamma=config.focal_loss_gamma)
    steps_per_epoch = len(data_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs * steps_per_epoch, eta_min=0)

    best_f1 = 0
    for epoch in range(config.epochs):
        print(f"====================Epoch {epoch + 1}/{config.epochs}====================")
        model.train()
        loss_meter = AverageMeter()

        global_step = epoch * len(data_loader)
        for step, sample in enumerate(data_loader):
            targets = sample['label'].cuda(device=device, non_blocking=True)

            pixels, mask, positions, extra = to_cuda(sample, device)
            outputs = forward_with_optional_source_reshaper(
                model,
                pixels,
                mask,
                positions,
                extra,
                labels=targets,
                source_feature_reshaper=source_feature_reshaper,
                apply_source_feature_reshaper=(source_feature_reshaper is not None and dataset_name == config.source),
            )
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_meter.update(loss.item(), n=config.batch_size)

            if step % config.log_step == 0:
                lr = optimizer.param_groups[0]["lr"]
                writer.add_scalar("train/loss", loss_meter.val, global_step + step)
                writer.add_scalar("train/lr", lr, global_step + step)

        model.eval()
        best_f1 = validation(
            best_f1,
            best_model_path,
            config,
            criterion,
            device,
            epoch,
            model,
            val_loader,
            writer,
            source_feature_reshaper=source_feature_reshaper,
            apply_source_feature_reshaper=False,
        )


def create_train_val_test_folds(datasets, num_folds, num_indices, val_ratio=0.1, test_ratio=0.2):
    folds = []
    for _ in range(num_folds):
        splits = {}
        for dataset in datasets:
            if type(num_indices) == dict:
                indices = list(range(num_indices[dataset]))
            else:
                indices = list(range(num_indices))
            n = len(indices)
            n_test = int(test_ratio * n)
            n_val = int(val_ratio * n)
            n_train = n - n_test - n_val

            random.shuffle(indices)

            train_indices = set(indices[:n_train])
            val_indices = set(indices[n_train:n_train + n_val])
            test_indices = set(indices[-n_test:])
            assert set.intersection(train_indices, val_indices, test_indices) == set()
            assert len(train_indices) + len(val_indices) + len(test_indices) == n

            splits[dataset] = {'train': train_indices, 'val': val_indices, 'test': test_indices}
        folds.append(splits)
    return folds


def save_results(metrics, config):
    out_dir = config.fold_dir
    metrics = deepcopy(metrics)
    conf_mat = metrics.pop('confusion_matrix')
    class_report = metrics.pop('classification_report')
    target_name = str(config.target).replace('/', '_')

    with open(os.path.join(out_dir, f'test_metrics_{target_name}.json'), 'w') as outfile:
        json.dump(metrics, outfile, indent=4)
    with open(os.path.join(out_dir, f'class_report_{target_name}.txt'), 'w') as outfile:
        outfile.write(str(class_report))
    pkl.dump(conf_mat, open(os.path.join(out_dir, f'conf_mat_{target_name}.pkl'), 'wb'))


def overall_performance(config):
    overall_metrics = defaultdict(list)
    target_name = str(config.target).replace("/", "_")

    cms = []
    for fold in range(config.num_folds):
        fold_dir = os.path.join(config.output_dir, f'fold_{fold}')
        test_metrics = json.load(open(os.path.join(fold_dir, f'test_metrics_{target_name}.json')))
        for metric, value in test_metrics.items():
            overall_metrics[metric].append(value)
        cm = pkl.load(open(os.path.join(fold_dir, f'conf_mat_{target_name}.pkl'), 'rb'))
        cms.append(cm)

    for i,row in enumerate(np.mean(cms, axis=0)):
        print(config.classes[i], row.astype(int))

    print(f'Overall result across {config.num_folds} folds:')
    print(overall_classification_report(cms, config.classes))
    for metric, values in overall_metrics.items():
        values = np.array(values)
        if metric == 'loss':
            print(f"{metric}: {np.mean(values):.4}±{np.std(values):.4}")
        else:
            values *= 100
            print(f"{metric}: {np.mean(values):.1f}±{np.std(values):.1f}")

    with open(os.path.join(config.output_dir, f'overall_{target_name}.json'), 'w') as file:
        file.write(json.dumps(overall_metrics, indent=4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Setup parameters
    parser.add_argument('--data_root', default='/data/user/DBL/timematch_data', type=str,
                        help='Path to datasets root directory')
    parser.add_argument('--num_blocks', default=100, type=int, help='Number of geographical blocks in dataset for splitting. Default 100.')

    available_tiles = ['denmark/32VNH/2017', 'france/30TXT/2017', 'france/31TCJ/2017', 'austria/33UVP/2017']

    parser.add_argument('--source', default='denmark/32VNH/2017', help='source dataset', choices=available_tiles)
    parser.add_argument('--target', default='france/30TXT/2017', help='target dataset', choices=available_tiles)
    parser.add_argument('--num_folds', default=1, type=int, help='Number of train/test folds for cross validation')
    parser.add_argument("--val_ratio", default=0.1, type=float,
                        help='Ratio of training data to use for validation. Default 10%.')
    parser.add_argument("--test_ratio", default=0.2, type=float,
                        help='Ratio of training data to use for testing. Default 20%.')
    parser.add_argument('--sample_pixels_val', type=bool_flag, default=True, help='speed up validation at the cost of randomness')
    parser.add_argument('--output_dir', default='outputs', help='Path to the folder where the results should be stored')
    parser.add_argument('-e', '--experiment_name', default=None, help='Name of the experiment')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers')
    parser.add_argument('--seed', default=1, type=int, help='Random seed')
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations')
    parser.add_argument('--log_step', default=10, type=int, help='Interval in batches between display of training metrics')
    parser.add_argument('--eval', action='store_true', help='run only evaluation')
    parser.add_argument('--overall', action='store_true', help='print overall results, if exists')
    parser.add_argument('--skip_final_test', default=False, type=bool_flag, help='skip final labeled test/eval stage after training; useful for warmup-only checkpoint-selection runs')
    parser.add_argument('--combine_spring_and_winter', default=False, type=bool_flag)
    parser.add_argument('--closed_set', default=False, type=bool_flag, help='exclude unknown / out-of-class parcels')

    # Training configuration
    parser.add_argument('--epochs', default=50, type=int, help='Number of epochs per fold')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay rate')
    parser.add_argument('--focal_loss_gamma', default=1.0, type=float, help='gamma value for focal loss')
    parser.add_argument('--num_pixels', default=64, type=int, help='Number of pixels to sample from the input sample')
    parser.add_argument('--seq_length', default=30, type=int, help='Number of time steps to sample from the input sample')
    parser.add_argument('--model', default='pseltae', choices=['psetae', 'pseltae', 'psetcnn', 'psegru'])
    parser.add_argument('--input_dim', default=10, type=int, help='Number of channels of input sample')
    parser.add_argument('--with_extra', default=False, type=bool_flag, help='whether to input extra geometric features to the PSE')
    parser.add_argument('--tensorboard_log_dir', default='runs')
    parser.add_argument('--train_on_target', default=False, action='store_true', help='supervised training on target for upper bound comparison')
    parser.add_argument('--source_checkpoint_epochs', default='', type=str, help='comma-separated source-only epochs to save extra checkpoints for later adaptation sweeps')
    parser.add_argument('--source_checkpoint_dirname', default='checkpoints', type=str, help='subdirectory name under each fold for extra source-only checkpoints')

    parser.add_argument('--with_shift_aug', default=False, type=bool_flag, help='whether to apply random temporal shift augmentation')
    parser.add_argument('--shift_aug_p', default=1.0, type=float, help='probability to apply temporal shift augmentation')
    parser.add_argument('--max_shift_aug', default=60, type=int, help='highest shift to apply for temporal shift augmentation')
    parser.add_argument(
        '--source_feature_reshaper',
        default='none',
        choices=['none', 'residual_temporal_conv'],
        help='source-only lightweight feature reshaper inserted between PSE and LTAE',
    )
    parser.add_argument(
        '--source_feature_reshaper_strength',
        default=0.10,
        type=float,
        help='initial residual strength for the source-only feature reshaper',
    )
    parser.add_argument(
        '--source_feature_reshaper_kernel_size',
        default=3,
        type=int,
        help='odd temporal kernel size used by the source-only feature reshaper',
    )
    parser.add_argument(
        '--source_feature_reshaper_reg_trade_off',
        default=0.05,
        type=float,
        help='weight for identity/stat-preserving regularization of the source-only feature reshaper',
    )
    parser.add_argument(
        '--source_feature_dual_path',
        default=False,
        type=bool_flag,
        help='use raw-source anchor + reshaped-source auxiliary dual-path training when source_feature_reshaper is enabled',
    )
    parser.add_argument(
        '--source_feature_dual_cls_trade_off',
        default=1.0,
        type=float,
        help='weight for reshaped-source classification loss in dual-path training',
    )
    parser.add_argument(
        '--source_feature_dual_relation_trade_off',
        default=0.05,
        type=float,
        help='weight for raw/reshaped source relation consistency in dual-path training',
    )
    parser.add_argument(
        '--source_phase_partition_mode',
        default='uniform',
        choices=['uniform', 'doy_gap', 'semantic_doy_gap', 'semantic_agglomerative'],
        help='phase partition mode for source phase compactness regularization',
    )
    parser.add_argument(
        '--source_segment_partition_mode',
        dest='source_segment_partition_mode',
        default=None,
        choices=['uniform', 'doy_gap', 'semantic_doy_gap', 'semantic_agglomerative'],
        help='segment partition mode alias for v2.4.0 temporal-segment abstraction; defaults to source_phase_partition_mode when omitted',
    )
    parser.add_argument(
        '--source_phase_count',
        default=5,
        type=int,
        help='phase count used only when source_phase_partition_mode=uniform',
    )
    parser.add_argument(
        '--source_segment_count',
        dest='source_segment_count',
        default=None,
        type=int,
        help='segment count alias for v2.4.0 temporal-segment abstraction; defaults to source_phase_count when omitted',
    )
    parser.add_argument(
        '--source_phase_gap_threshold',
        default=45,
        type=int,
        help='split phase candidates when consecutive source DOYs differ by more than this threshold',
    )
    parser.add_argument(
        '--source_phase_min_points',
        default=3,
        type=int,
        help='minimum number of source-domain time points required for a phase segment',
    )
    parser.add_argument(
        '--source_phase_max_points',
        default=8,
        type=int,
        help='maximum number of source-domain time points before a phase segment is split',
    )
    parser.add_argument(
        '--source_phase_max_span',
        default=120,
        type=int,
        help='maximum DOY span for one source phase before it is split',
    )
    parser.add_argument(
        '--source_phase_min_sample_points',
        default=2,
        type=int,
        help='minimum sampled time points required for one sample to contribute to a phase loss',
    )
    parser.add_argument(
        '--source_segment_semantic_quantile',
        default=0.75,
        type=float,
        help='quantile threshold for selecting semantic split candidates in semantic_doy_gap segment partition mode',
    )
    parser.add_argument(
        '--source_segment_semantic_max_samples_per_class',
        default=128,
        type=int,
        help='maximum number of source samples per class used to estimate semantic boundary scores',
    )
    parser.add_argument(
        '--source_segment_semantic_curvature_trade_off',
        default=0.5,
        type=float,
        help='trade-off weight for curvature when constructing semantic boundary scores in semantic_doy_gap mode',
    )
    parser.add_argument(
        '--source_segment_semantic_energy_trade_off',
        default=0.25,
        type=float,
        help='trade-off weight for local energy change when constructing semantic boundary scores in semantic_doy_gap mode',
    )
    parser.add_argument(
        '--source_segment_semantic_similarity_trade_off',
        default=0.25,
        type=float,
        help='trade-off weight for local cosine dissimilarity when constructing semantic boundary scores in semantic_doy_gap mode',
    )
    parser.add_argument(
        '--source_segment_semantic_max_extra_cuts_per_base',
        default=2,
        type=int,
        help='maximum number of semantic split candidates inserted inside each gap-based base segment',
    )
    parser.add_argument(
        '--source_segment_semantic_merge_boundary_trade_off',
        default=0.5,
        type=float,
        help='trade-off weight for preserving high-score boundaries during semantic agglomerative segmentation',
    )
    parser.add_argument(
        '--source_segment_semantic_aggl_min_points',
        default=3,
        type=int,
        help='minimum preferred segment length used when allocating semantic agglomerative segment capacity',
    )
    parser.add_argument(
        '--source_segment_semantic_aggl_target_slack',
        default=1,
        type=int,
        help='allow semantic agglomerative segmentation to end within target_count +/- this slack',
    )
    parser.add_argument(
        '--source_segment_semantic_aggl_merge_cost_tolerance',
        default=1.15,
        type=float,
        help='relative tolerance on initial merge costs when deciding whether semantic agglomerative merging should continue past the upper target count',
    )
    parser.add_argument(
        '--source_segment_semantic_aggl_dynamics_trade_off',
        default=0.35,
        type=float,
        help='trade-off weight for local dynamics similarity in semantic agglomerative merge cost',
    )
    parser.add_argument(
        '--source_structure_loss_version',
        default='compactness',
        choices=['compactness', 'multi_component', 'profiled_components', 'trend_residual', 'trend_seasonal_residual', 'segment_trend_residual', 'segment_transition_residual', 'segment_transition_semantic', 'segment_boundary_window_residual'],
        help='source-side structural loss version: compactness, v2.3.2 multi-component, v2.3.3 profiled, v2.3.4 trend-residual, v2.3.5 trend-seasonal-residual, v2.4.0 segment-trend-residual, v2.4.1 segment-transition-residual, v2.4.2 semantic-segment transition residual, or v2.4.3 boundary-window segment transition residual',
    )
    parser.add_argument(
        '--source_structure_intra_trade_off',
        default=1.0,
        type=float,
        help='component weight for intra-phase compactness in source structure loss',
    )
    parser.add_argument(
        '--source_structure_amplitude_trade_off',
        default=0.25,
        type=float,
        help='component weight for amplitude-spread suppression in v2.3.2 multi-component source structure loss',
    )
    parser.add_argument(
        '--source_structure_interphase_trade_off',
        default=0.25,
        type=float,
        help='component weight for adjacent inter-phase smoothness in v2.3.2 multi-component source structure loss',
    )
    parser.add_argument(
        '--source_structure_shape_trade_off',
        default=0.15,
        type=float,
        help='shape-regularization weight in profiled source structure loss variants',
    )
    parser.add_argument(
        '--source_structure_trend_trade_off',
        default=0.05,
        type=float,
        help='trend-smoothness regularization weight in trend-residual source structure loss variants',
    )
    parser.add_argument(
        '--source_structure_season_trade_off',
        default=0.02,
        type=float,
        help='seasonal-pattern regularization weight in trend-seasonal-residual source structure loss variants',
    )
    parser.add_argument(
        '--source_structure_segment_inter_trade_off',
        default=0.02,
        type=float,
        help='adjacent inter-segment transition regularization weight in v2.4.1 segment-transition-residual source structure loss',
    )
    parser.add_argument(
        '--source_structure_boundary_window_trade_off',
        default=0.02,
        type=float,
        help='boundary-centered local window regularization weight in v2.4.3 boundary-window segment source structure loss',
    )
    parser.add_argument(
        '--source_structure_boundary_window_size',
        default=2,
        type=int,
        help='half-window size per side used for v2.4.3 boundary-centered local segment transition windows',
    )
    # Specific parameters for each training method
    subparsers = parser.add_subparsers(dest='method')

    # DANN + CDAN
    dann = subparsers.add_parser('dann')
    dann.add_argument('--adv_loss', type=str, default='DANN', choices=['DANN', 'CDAN', 'CDAN+E'])
    dann.add_argument('--use_default_optim', type=bool_flag, default=True, help="whether to use default optimizer")
    dann.add_argument('--weights', type=str, help='path to source trained model weights')
    dann.add_argument("--steps_per_epoch", type=int, default=500, help='n steps per epoch')
    dann.add_argument('--epochs', default=20, type=int, help='Number of epochs per fold')
    dann.add_argument("--trade_off", default=1.0, type=float, help='weight of adversarial loss')
    dann.add_argument('--lr', default=0.001, type=float, help='Learning rate')

    # MMD loss (DAN)
    mmd = subparsers.add_parser('mmd')
    mmd.add_argument('--use_default_optim', type=bool_flag, default=True, help="whether to use default optimizer")
    mmd.add_argument('--weights', type=str, help='path to source trained model weights')
    mmd.add_argument("--steps_per_epoch", type=int, default=500, help='n steps per epoch')
    mmd.add_argument('--epochs', default=20, type=int, help='Number of epochs per fold')
    mmd.add_argument("--trade_off", default=1.0, type=float, help='weight of adversarial loss')
    mmd.add_argument('--lr', default=0.001, type=float, help='Learning rate')

    # JUMBOT
    jumbot = subparsers.add_parser('jumbot')
    jumbot.add_argument('--weights', type=str, help='path to source trained model weights')
    jumbot.add_argument("--steps_per_epoch", type=int, default=500, help='n steps per epoch')
    jumbot.add_argument('--epochs', default=20, type=int, help='Number of epochs per fold')
    jumbot.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    jumbot.add_argument('--eta1', default=0.01, type=float, help='feature comparison coefficient')
    jumbot.add_argument('--eta2', default=1.0, type=float, help='label comparison coefficient')
    jumbot.add_argument('--epsilon', default=0.01, type=float, help='marginal coefficient')
    jumbot.add_argument('--tau', default=0.5, type=float, help='entropic regularization')

    # ALDA
    alda = subparsers.add_parser('alda')
    alda.add_argument('--use_default_optim', type=bool_flag, default=True, help="whether to use default optimizer")
    alda.add_argument('--weights', type=str, help='path to source trained model weights')
    alda.add_argument("--steps_per_epoch", type=int, default=500, help='n steps per epoch')
    alda.add_argument('--epochs', default=20, type=int, help='Number of epochs per fold')
    alda.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    alda.add_argument("--trade_off", default=1.0, type=float, help='weight of adversarial loss')
    alda.add_argument("--pseudo_threshold", default=0.7, type=float, help='confidence threshold for assigning pseudo labels')


    # TimeMatch
    timematch = subparsers.add_parser('timematch')
    timematch.add_argument('--weights', type=str, help='path to source trained model weights')
    timematch.add_argument('--weights_checkpoint', type=str, default='model.pt', help='checkpoint filename under weights/fold_k or absolute checkpoint path')
    timematch.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
    timematch.add_argument("--pseudo_threshold", default=0.9, type=float, help='confidence threshold for assigning pseudo labels')
    timematch.add_argument("--ema_decay", default=0.9999, type=float, help='decay rate for mean teacher')
    timematch.add_argument("--trade_off", type=float, default=2.0, help='weight for unsupervised loss')
    timematch.add_argument("--estimate_shift", type=bool_flag, default=True, help='whether to account for temporal shift')
    timematch.add_argument('--epochs', default=20, type=int, help='Number of epochs per fold')
    timematch.add_argument("--steps_per_epoch", type=int, default=500, help='n steps per epoch')
    timematch.add_argument("--balance_source", type=bool_flag, default=True, help='class balanced batches for source')
    timematch.add_argument("--use_focal_loss", type=bool_flag, default=True, help='use focal loss or cross entropy')
    timematch.add_argument("--shift_source", type=bool_flag, default=True, help='whether to apply temporal shift to source data')
    timematch.add_argument("--sample_size", type=int, default=100, help='number of batches to sample for estimating shift')
    timematch.add_argument("--max_temporal_shift", type=int, default=60, help='maximum temporal shift to consider')
    timematch.add_argument("--domain_specific_bn", type=bool_flag, default=True, help='whether to use domain specific batch normalization')
    timematch.add_argument("--shift_estimator", type=str, default='AM', choices=['AM', 'IS', 'ACC', 'ENT'])
    timematch.add_argument('--run_validation', default=True, action='store_true', help='whether to run validation each epoch')
    timematch.add_argument('--disable_validation_in_timematch', default=False, type=bool_flag, help='disable labeled validation during TimeMatch; useful for warmup-only checkpoint-selection runs')
    timematch.add_argument("--output_student", type=bool_flag, default=True, help='output student or teacher')
    timematch.add_argument('--selection_metrics_out', type=str, default=None, help='optional JSON path for saving target-side unsupervised checkpoint-selection metrics after training')
    timematch.add_argument('--selection_score_coverage_weight', type=float, default=0.25, help='weight for high-confidence pseudo-label coverage in target-aware checkpoint selection score')
    timematch.add_argument('--selection_score_confidence_weight', type=float, default=0.15, help='weight for mean teacher confidence in target-aware checkpoint selection score')
    timematch.add_argument('--selection_score_agreement_weight', type=float, default=0.15, help='weight for teacher-student agreement in target-aware checkpoint selection score')
    timematch.add_argument('--selection_score_entropy_weight', type=float, default=0.10, help='penalty weight for normalized target prediction entropy in target-aware checkpoint selection score')
    timematch.add_argument('--selection_score_class_balance_weight', type=float, default=0.45, help='weight for pseudo-label class-distribution health in v2.5.3 checkpoint selection')
    timematch.add_argument('--selection_score_source_prior_weight', type=float, default=0.25, help='penalty weight for pseudo-label/source-prior JS divergence in v2.5.3 checkpoint selection')
    timematch.add_argument('--selection_score_shift_stability_weight', type=float, default=0.20, help='weight for shift trajectory stability in v2.5.3 checkpoint selection')
    timematch.add_argument('--selection_metric_batches', type=int, default=200, help='number of target batches used to compute final unsupervised selection metrics')
    timematch.add_argument('--selection_score_mode', type=str, default='temporal_perturbation', choices=['legacy', 'temporal_perturbation', 'temporal_perturbation_trajectory', 'temporal_perturbation_late_filter'], help='checkpoint selection score mode')
    timematch.add_argument('--selection_time_mask_p', type=float, default=0.15, help='probability of replacing a time step by the sequence mean when computing perturbation consistency')
    timematch.add_argument('--selection_temporal_jitter', type=int, default=3, help='maximum absolute temporal position jitter for perturbation consistency')
    timematch.add_argument('--selection_value_noise_std', type=float, default=0.03, help='relative Gaussian value noise level for perturbation consistency')
    timematch.add_argument('--selection_perturbation_weight', type=float, default=1.0, help='weight for perturbation consistency in temporal_perturbation selection score')
    timematch.add_argument('--selection_collapse_penalty_weight', type=float, default=0.35, help='penalty weight for pseudo-label class collapse in temporal_perturbation selection score')
    timematch.add_argument('--selection_trajectory_alpha', type=float, default=0.30, help='penalty strength for late warmup perturbation-consistency gains in temporal_perturbation_trajectory mode')
    timematch.add_argument('--selection_late_gain_threshold', type=float, default=0.20, help='late-gain ratio threshold before applying v2.5.3d trajectory soft filter')

    # Source-only + source phase compactness regularization
    sourcephasecompact = subparsers.add_parser('sourcephasecompact')

    cfg = parser.parse_args()


    # Setup folders based on name
    if cfg.experiment_name is not None:
        cfg.tensorboard_log_dir = os.path.join(cfg.tensorboard_log_dir, cfg.experiment_name)
        cfg.output_dir = os.path.join(cfg.output_dir, cfg.experiment_name)

    os.makedirs(cfg.output_dir, exist_ok=True)
    for fold in range(cfg.num_folds):
        os.makedirs(os.path.join(cfg.output_dir, 'fold_{}'.format(fold)), exist_ok=True)


    # write training config to file
    if not cfg.eval:
        with open(os.path.join(cfg.output_dir, 'train_config.json'), 'w') as f:
            f.write(json.dumps(vars(cfg), indent=4))
    print(cfg)
    main(cfg)

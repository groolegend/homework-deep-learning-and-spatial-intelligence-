import argparse

from src.data import load_eurosat, train_val_test_split
from src.model import MLP3
from src.test import load_model_weights, test_model
from src.train import train_model


def build_model(in_dim, out_dim, cfg):
    return MLP3(
        in_dim,
        cfg['h1'],
        cfg['h2'],
        out_dim,
        act=cfg['act'],
        dropout_p=cfg['dropout_p'],
        layernorm=cfg['layernorm'],
    )


def get_config(ckpt_path=None):
    cfg = {
        'h1': 1024,
        'h2': 512,
        'act': 'relu',
        'dropout_p': 0.15,
        'layernorm': True,

        'lr': 0.004,
        'max_lr': 0.015,
        'warmup_frac': 0.1,
        'lr_schedule': 'onecycle',
        'lr_decay': 0.99,
        'weight_decay': 5e-5,
        'momentum': 0.9,
        'epochs': 160,
        'batch_size': 128,
        'augment': True,
        'label_smoothing': 0.05,
        'early_stop_patience': 25,

        'save_path': 'checkpoints/best_single_strong.npz',
        'plot_path': 'checkpoints/training_curve.png',
        'history_path': 'checkpoints/training_history.npz',
    }

    if ckpt_path is not None:
        cfg['save_path'] = ckpt_path

    return cfg


def load_data(data_root='EuroSAT_RGB'):
    X, y, class_names = load_eurosat(data_root)
    print(f'Loaded dataset: X={X.shape}, y={y.shape}, classes={class_names}')

    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(
        X, y, train_ratio=0.7, val_ratio=0.15, seed=42
    )

    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-6

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    return X, y, class_names, X_train, y_train, X_val, y_val, X_test, y_test


def train(args):
    cfg = get_config(args.ckpt)

    (
        X, y, class_names,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
    ) = load_data(args.data_root)

    in_dim = X.shape[1]
    out_dim = len(class_names)

    print('\nRunning train config:')
    print(cfg)

    model = build_model(in_dim, out_dim, cfg)

    train_model(
        model,
        X_train, y_train,
        X_val, y_val,
        epochs=cfg['epochs'],
        batch_size=cfg['batch_size'],
        lr=cfg['lr'],
        lr_decay=cfg['lr_decay'],
        lr_schedule=cfg['lr_schedule'],
        max_lr=cfg['max_lr'],
        warmup_frac=cfg['warmup_frac'],
        weight_decay=cfg['weight_decay'],
        momentum=cfg['momentum'],
        augment=cfg['augment'],
        label_smoothing=cfg['label_smoothing'],
        early_stop_patience=cfg['early_stop_patience'],
        plot_path=cfg['plot_path'],
        history_path=cfg['history_path'],
        save_path=cfg['save_path'],
    )

    print(f'\nLoading best checkpoint from: {cfg["save_path"]}')
    load_model_weights(model, cfg['save_path'])

    print('\nTesting best model:')
    test_model(model, X_test, y_test, num_classes=out_dim)


def eval_model(args):
    cfg = get_config(args.ckpt)

    (
        X, y, class_names,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
    ) = load_data(args.data_root)

    in_dim = X.shape[1]
    out_dim = len(class_names)

    model = build_model(in_dim, out_dim, cfg)

    print(f'\nLoading checkpoint from: {cfg["save_path"]}')
    load_model_weights(model, cfg['save_path'])

    print('\nEvaluating model:')
    test_model(model, X_test, y_test, num_classes=out_dim)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'mode',
        choices=['train', 'eval'],
        help='train: train model; eval: evaluate checkpoint',
    )

    parser.add_argument(
        '--ckpt',
        type=str,
        default='checkpoints/best_single_strong.npz',
        help='checkpoint path for saving/loading model',
    )

    parser.add_argument(
        '--data_root',
        type=str,
        default='EuroSAT_RGB',
        help='dataset root directory',
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        eval_model(args)


if __name__ == '__main__':
    main()
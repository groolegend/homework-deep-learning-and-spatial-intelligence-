import itertools
import random
from .model import MLP3
from .train import train_model


def grid_search(
    X_train, y_train, X_val, y_val, in_dim, out_dim,
    lrs, hidden_pairs, decays, wds, activations, dropouts, momentums,
    epochs=15, batch_size=64, save_dir='checkpoints'
):
    results = []
    for lr, (h1, h2), decay, wd, act, do, mom in itertools.product(lrs, hidden_pairs, decays, wds, activations, dropouts, momentums):
        name = f'{save_dir}/best_lr{lr}_h{h1}-{h2}_d{decay}_wd{wd}_{act}_do{do}_m{mom}.npz'
        model = MLP3(in_dim=in_dim, h1=h1, h2=h2, out_dim=out_dim, act=act, dropout_p=do)
        best_val, best_ep = train_model(
            model, X_train, y_train, X_val, y_val,
            epochs=epochs, batch_size=batch_size,
            lr=lr, lr_decay=decay, weight_decay=wd, momentum=mom, save_path=name
        )
        results.append({
            'lr': lr, 'h1': h1, 'h2': h2, 'decay': decay, 'wd': wd, 'act': act, 'dropout': do, 'momentum': mom,
            'best_val_acc': best_val, 'best_epoch': best_ep, 'ckpt': name
        })
    results.sort(key=lambda x: x['best_val_acc'], reverse=True)
    return results


def random_search(
    X_train, y_train, X_val, y_val, in_dim, out_dim,
    lr_choices, hidden_choices, decay_choices, wd_choices, act_choices,
    trials=8, epochs=15, batch_size=64, save_dir='checkpoints'
):
    results = []
    for _ in range(trials):
        lr = random.choice(lr_choices)
        h1, h2 = random.choice(hidden_choices)
        decay = random.choice(decay_choices)
        wd = random.choice(wd_choices)
        act = random.choice(act_choices)

        name = f'{save_dir}/best_lr{lr}_h{h1}-{h2}_d{decay}_wd{wd}_{act}.npz'
        model = MLP3(in_dim=in_dim, h1=h1, h2=h2, out_dim=out_dim, act=act)
        best_val, best_ep = train_model(
            model, X_train, y_train, X_val, y_val,
            epochs=epochs, batch_size=batch_size,
            lr=lr, lr_decay=decay, weight_decay=wd, save_path=name
        )
        results.append({
            'lr': lr, 'h1': h1, 'h2': h2, 'decay': decay, 'wd': wd, 'act': act,
            'best_val_acc': best_val, 'best_epoch': best_ep, 'ckpt': name
        })

    results.sort(key=lambda x: x['best_val_acc'], reverse=True)
    return results

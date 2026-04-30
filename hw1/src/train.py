import os
import numpy as np
from .autograd import Tensor
from .losses import cross_entropy_with_logits, l2_regularization
from .optim import SGD


def _augment_batch_flat(xb, img_hw=64, channels=3):
    # xb: (B, H*W*C) float32
    bsz = xb.shape[0]
    x = xb.reshape(bsz, img_hw, img_hw, channels)
    out = x.copy()
    for i in range(bsz):
        r = np.random.rand()
        if r < 0.5:
            out[i] = out[i, :, ::-1, :]  # horizontal flip
        if np.random.rand() < 0.5:
            out[i] = out[i, ::-1, :, :]  # vertical flip
        k = np.random.randint(0, 4)
        if k:
            out[i] = np.rot90(out[i], k=k, axes=(0, 1))
        # light color jitter + gaussian noise (works well for EuroSAT)
        if np.random.rand() < 0.7:
            scale = np.random.uniform(0.9, 1.1)
            bias = np.random.uniform(-0.05, 0.05)
            out[i] = np.clip(out[i] * scale + bias, -5.0, 5.0)
        if np.random.rand() < 0.5:
            out[i] = out[i] + np.random.randn(*out[i].shape).astype(np.float32) * 0.02
    return out.reshape(bsz, -1)


def accuracy_from_logits(logits_np, y):
    pred = np.argmax(logits_np, axis=1)
    return float((pred == y).mean())


def evaluate(model, X, y, batch_size=256):
    from .data import iter_batches
    model.eval()
    all_logits, all_y = [], []
    for xb, yb in iter_batches(X, y, batch_size=batch_size, shuffle=False):
        xt = Tensor(xb, requires_grad=False)
        logits = model(xt).data
        all_logits.append(logits)
        all_y.append(yb)
    logits_np = np.concatenate(all_logits, axis=0)
    y_np = np.concatenate(all_y, axis=0)
    return accuracy_from_logits(logits_np, y_np)


def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=20,
    batch_size=64,
    lr=0.05,
    lr_decay=0.95,
    lr_schedule='exp',  # 'exp' or 'onecycle'
    max_lr=None,
    warmup_frac=0.1,
    weight_decay=1e-4,
    momentum=0.0,
    augment=False,
    label_smoothing=0.0,
    early_stop_patience=None,
    plot_path=None,
    history_path=None,
    save_path='best_model.npz',
):
    optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    optimizer.nesterov = True

    best_val_acc = -1.0
    best_epoch = -1
    no_improve = 0

    history = {
        'epoch': [],
        'lr': [],
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
    }

    from .data import iter_batches

    total_steps = int(np.ceil(len(X_train) / batch_size)) * epochs
    warmup_steps = int(total_steps * float(warmup_frac))
    step_id = 0

    def _onecycle_lr(step):
        # linear warmup to max_lr, then cosine to min_lr (= lr/25)
        base_lr = lr
        peak = float(max_lr if max_lr is not None else lr)
        min_lr = base_lr / 25.0
        if warmup_steps > 0 and step < warmup_steps:
            return base_lr + (peak - base_lr) * (step / warmup_steps)
        t = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
        return min_lr + 0.5 * (peak - min_lr) * (1.0 + np.cos(np.pi * t))

    for ep in range(1, epochs + 1):
        model.train()

        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        last_lr = optimizer.lr

        for xb, yb in iter_batches(X_train, y_train, batch_size=batch_size, shuffle=True):
            if lr_schedule == 'exp':
                optimizer.lr = lr * (lr_decay ** (ep - 1))
            elif lr_schedule == 'onecycle':
                optimizer.lr = float(_onecycle_lr(step_id))
            step_id += 1
            last_lr = optimizer.lr

            if augment:
                xb = _augment_batch_flat(xb, img_hw=64, channels=3)
            xt = Tensor(xb, requires_grad=False)
            logits = model(xt)
            ce = cross_entropy_with_logits(logits, yb, label_smoothing=label_smoothing)
            l2 = l2_regularization(model, weight_decay)
            loss = ce + Tensor(np.array(l2, dtype=np.float32), requires_grad=False)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.data) * len(xb)
            pred = np.argmax(logits.data, axis=1)
            train_correct += int((pred == yb).sum())
            train_total += len(xb)

        train_acc = train_correct / train_total
        train_loss = train_loss_sum / train_total
        val_acc = evaluate(model, X_val, y_val)

        history['epoch'].append(ep)
        history['lr'].append(float(last_lr))
        history['train_loss'].append(float(train_loss))
        history['train_acc'].append(float(train_acc))
        history['val_acc'].append(float(val_acc))

        print(
            f'[Epoch {ep:02d}] lr={optimizer.lr:.6f} '
            f'train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}'
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = ep
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            np.savez(save_path, **model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if early_stop_patience is not None and no_improve >= int(early_stop_patience):
                print(f'Early stopping at epoch={ep} (patience={early_stop_patience})')
                break

        if plot_path is not None:
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

                os.makedirs(os.path.dirname(plot_path) or '.', exist_ok=True)
                fig, ax1 = plt.subplots(figsize=(8, 4.5))
                ax1.plot(history['epoch'], history['train_loss'], label='train_loss', color='tab:blue')
                ax1.set_xlabel('epoch')
                ax1.set_ylabel('train_loss', color='tab:blue')
                ax1.tick_params(axis='y', labelcolor='tab:blue')
                ax1.grid(True, alpha=0.25)

                ax2 = ax1.twinx()
                ax2.plot(history['epoch'], history['val_acc'], label='val_acc', color='tab:green')
                ax2.plot(history['epoch'], history['train_acc'], label='train_acc', color='tab:orange', alpha=0.6)
                ax2.set_ylabel('accuracy', color='tab:green')
                ax2.tick_params(axis='y', labelcolor='tab:green')
                ax2.set_ylim(0.0, 1.0)

                title = f'best_val={best_val_acc:.4f} @ epoch={best_epoch}'
                ax1.set_title(title)

                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

                fig.tight_layout()
                fig.savefig(plot_path, dpi=160)
                plt.close(fig)
            except Exception as e:
                print(f'[warn] plot failed: {e}')

        if history_path is not None:
            try:
                os.makedirs(os.path.dirname(history_path) or '.', exist_ok=True)
                np.savez(
                    history_path,
                    epoch=np.array(history['epoch'], dtype=np.int64),
                    lr=np.array(history['lr'], dtype=np.float32),
                    train_loss=np.array(history['train_loss'], dtype=np.float32),
                    train_acc=np.array(history['train_acc'], dtype=np.float32),
                    val_acc=np.array(history['val_acc'], dtype=np.float32),
                    best_val_acc=np.array(best_val_acc, dtype=np.float32),
                    best_epoch=np.array(best_epoch, dtype=np.int64),
                )
            except Exception as e:
                print(f'[warn] save history failed: {e}')

    print(f'Best val acc={best_val_acc:.4f} at epoch={best_epoch}')
    return best_val_acc, best_epoch

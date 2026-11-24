'''Live training dashboard for pb_train.py: reads train_log.jsonl and draws the curves.
Chart text is English on purpose (no CJK font for matplotlib here). One axis per chart, fixed blue / orange order,
2px lines, grey for the reference, direct label on the last point.'''
import json, os, time
import matplotlib.pyplot as plt

LOG = '/home/dev/KAP/data/pb_dspark/train_log.jsonl'
SURF, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
BLUE, ORANGE = '#2a78d6', '#eb6834'


def read(path=LOG):
    out = []
    if os.path.exists(path):
        with open(path) as f:
            for ln in f:
                try:
                    out.append(json.loads(ln))
                except ValueError:
                    pass                      # the last line may be half written
    return out


def _style(ax, title, ylabel, xlabel='epoch'):
    ax.set_facecolor(SURF)
    ax.set_title(title, loc='left', fontsize=10, color=INK)
    ax.set_ylabel(ylabel, fontsize=8, color=INK2)
    ax.set_xlabel(xlabel, fontsize=8, color=INK2)
    ax.grid(axis='y', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def _line(ax, x, y, color, label, marker=None):
    if not x:
        return
    ax.plot(x, y, color=color, lw=2, label=label, marker=marker, ms=8, mec=SURF, mew=2)
    prev = getattr(ax, '_end_labels', [])
    lo, hi = ax.get_ylim()
    dy = 0
    for p in prev:
        if abs(p - y[-1]) < 0.06 * (hi - lo):      # two end labels would touch: push this one away from the other
            dy = 6 if y[-1] >= p else -6
    ax._end_labels = prev + [y[-1]]
    ax.annotate(f"{y[-1]:.3f}", (x[-1], y[-1]), xytext=(6, dy), textcoords='offset points', fontsize=8, color=INK2, va='center')


def _pad(ax):
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi + 0.2 * (hi - lo))


def progress(recs):
    '''Latest config wins: resuming for more epochs appends a new config record.
    Train records before a resume keep their own (smaller) step numbers, so only the
    tail after the last config is used for the pace estimate.'''
    cfg = next((r for r in reversed(recs) if r['t'] == 'config'), None)
    tr = [r for r in recs if r['t'] == 'train']
    if not (cfg and tr):
        return 'waiting for training logs...'
    last = tr[-1]
    left = (cfg['total_steps'] - last['step']) * last['ms'] / 1000
    sps = last['step'] - cfg.get('resumed_from', 0)
    return (f"step {last['step']}/{cfg['total_steps']}  epoch {last['epoch']:.2f}/{cfg['epochs']}  "
            f"{last['ms']:.0f} ms/step  ETA {left / 3600:.1f} h")


def verdict(recs):
    va = [r for r in recs if r['t'] == 'val']
    out = [progress(recs)]
    if len(va) < 2:
        out.append('只有起点评测（step 0），还没有可比较的验证点' if va else '还没有验证结果')
    else:
        f, p, l = va[0], va[-2], va[-1]
        w = lambda d: '变好' if d > 0.01 else ('变差' if d < -0.01 else '基本持平')
        out.append(f"val tau  起点 {f['tau']:.3f} -> 现在 {l['tau']:.3f} ({l['tau'] - f['tau']:+.3f}，{w(l['tau'] - f['tau'])})；"
                   f"较上一次验证 {l['tau'] - p['tau']:+.3f}（{w(l['tau'] - p['tau'])}）；最佳 {max(r['tau'] for r in va):.3f}")
        if len(va) >= 4 and all(va[-i]['tau'] < va[-i - 1]['tau'] for i in (1, 2, 3)):
            out.append('警告：验证 tau 已连续 3 次下降，可能过拟合或学习率偏高')
    done = [r for r in recs if r['t'] == 'done']
    if done and recs and recs[-1]['t'] == 'done':
        out.append(f"训练结束；Target 权重文件 sha256 前后一致: {done[-1]['target_unchanged']}")
    return chr(10).join(out)


def dashboard(recs):
    tr = [r for r in recs if r['t'] == 'train']
    va = [r for r in recs if r['t'] == 'val']
    if not tr and not va:
        return None
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.6), facecolor=SURF)
    ex, vx = [r['epoch'] for r in tr], [r['epoch'] for r in va]

    def pair(ax, key, title, ylabel):
        _style(ax, title, ylabel)
        _line(ax, ex, [r[key] for r in tr], BLUE, 'train')
        _line(ax, vx, [r[key] for r in va], ORANGE, 'val', 'o')
        _pad(ax)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2)

    _style(axs[0, 0], 'Training loss (0.1 CE + 0.9 TV + 1.0 BCE, position-weighted)', 'loss')
    _line(axs[0, 0], ex, [r['loss'] for r in tr], BLUE, 'train')
    _pad(axs[0, 0])
    pair(axs[0, 1], 'tau', 'tau = E[n] + 1 (higher is better)', 'tokens per round')
    pair(axs[0, 2], 'tv', 'TV distance (lower is better)', 'sum |q - p|')
    pair(axs[1, 0], 'ce', 'Cross-entropy on sampled tokens (lower is better)', 'nats')

    ax = axs[1, 1]
    _style(ax, 'Per-slot acceptance alpha on val (higher is better)', 'alpha = sum min(p, q)', 'draft slot')
    if va:
        s = list(range(1, len(va[0]['alpha']) + 1))
        ax.plot(s, va[0]['alpha'], color=MUTED, lw=2, ls='--', label='start (step %d)' % va[0]['step'])
        if len(va) > 1:
            ax.plot(s, va[-1]['alpha'], color=BLUE, lw=2, marker='o', ms=8, mec=SURF, mew=2, label='latest (step %d)' % va[-1]['step'])
        ax.set_ylim(0, 1)
        ax.set_xticks(s)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2)

    ax = axs[1, 2]
    _style(ax, 'Confidence head vs actual acceptance (latest val)', 'probability', 'draft slot')
    if va:
        s = list(range(1, len(va[-1]['alpha']) + 1))
        ax.plot(s, va[-1]['alpha'], color=BLUE, lw=2, marker='o', ms=8, mec=SURF, mew=2, label='actual alpha')
        ax.plot(s, va[-1]['conf'], color=ORANGE, lw=2, marker='o', ms=8, mec=SURF, mew=2, label='confidence head')
        ax.set_ylim(0, 1)
        ax.set_xticks(s)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2)

    fig.suptitle(progress(recs), x=0.01, ha='left', fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def live(interval=30, once=False, path=LOG):
    from IPython.display import clear_output, display
    while True:
        recs = read(path)
        clear_output(wait=True)
        print(verdict(recs))
        fig = dashboard(recs)
        if fig is not None:
            display(fig)
            plt.close(fig)
        if once or (recs and recs[-1]['t'] == 'done'):
            break
        time.sleep(interval)

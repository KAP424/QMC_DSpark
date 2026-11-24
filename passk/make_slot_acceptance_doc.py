"""Build the per-slot acceptance figure + markdown tables for the doc.

Data sources:
  maze, countdown : exp_group_coupling.json (100 prompts x 8 rollouts x 5 modes;
                    per-slot counts derived from the per-round accepted-k lists)
  sudoku, 1d_arc  : exp_slot_acceptance.json (40 prompts x 8 rollouts x 5 modes)

Output:
  /home/dev/KAP/doc/figures/slot_acceptance.png
  stdout: markdown tables to paste into the doc
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

ND = 7
BASE = '/home/dev/KAP/passk'
DOC = '/home/dev/KAP/doc'

MODES = ['stratified', 'iid', 'stratified_fixed']
LABELS = {
    'stratified':       'inv-CDF coupled (permuted strata)',
    'iid':              'inv-CDF iid',
    'stratified_fixed': 'inv-CDF coupled, fixed strata (pre-fix)',
}
# validated categorical palette (dataviz reference instance, light mode)
STYLE = {
    'stratified':       dict(color='#2a78d6', ls='-',  marker='o', ms=5.5),
    'iid':              dict(color='#eb6834', ls='-',  marker='s', ms=5.5),
    'stratified_fixed': dict(color='#e87ba4', ls=':',  marker='x', ms=6.0),
}
INK, MUTE, GRID = '#0b0b0b', '#898781', '#e1e0d9'
SURFACE, BASELINE = '#fcfcfb', '#c3c2b7'


def counts_from_accepted(accepted_lists):
    reach = [0] * ND
    accept = [0] * ND
    for ks in accepted_lists:
        for k in ks:
            for j in range(1, ND + 1):
                if k >= j - 1:
                    reach[j - 1] += 1
                    if k >= j:
                        accept[j - 1] += 1
    return reach, accept


def alpha_from_counts(reach, accept):
    return [a / r if r else None for a, r in zip(accept, reach)]


# ---- assemble per-domain per-mode slot stats ---------------------------------
slot = {}
for fn in ('exp_slot_acceptance_mc.json', 'exp_slot_acceptance.json'):
    try:
        sa = json.load(open(f'{BASE}/{fn}'))
    except FileNotFoundError:
        print(f'[warn] {fn} missing')
        continue
    for ds in sa:
        slot.setdefault(ds, {})
        for m in MODES:
            if m in sa[ds]:
                slot[ds][m] = sa[ds][m]

DS_ORDER = [d for d in ('maze', 'sudoku', 'countdown', '1d_arc') if d in slot]
DS_LABELS = {'maze': 'Maze', 'sudoku': 'Sudoku', 'countdown': 'Countdown', '1d_arc': '1D-ARC'}

# ---- figure ------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 7.2), dpi=150)
fig.patch.set_facecolor(SURFACE)
axes = axes.flatten()
X = list(range(1, ND + 1))

for ax, ds in zip(axes, DS_ORDER):
    ax.set_facecolor(SURFACE)
    for m in MODES:
        if m not in slot.get(ds, {}):
            continue
        a = slot[ds][m]['alpha']
        ys = [(v * 100 if v is not None else float('nan')) for v in a]
        st = STYLE[m]
        ax.plot(X, ys, color=st['color'], lw=2, ls=st['ls'],
                marker=st['marker'], markersize=st['ms'], markeredgewidth=1.4,
                label=LABELS[m], zorder=3)
    ax.set_title(f'{DS_LABELS[ds]}', fontsize=12, color=INK,
                 fontweight='semibold', loc='left', pad=6)
    ax.set_xlabel('draft token position', fontsize=9, color=MUTE)
    ax.set_ylabel('acceptance rate (%)', fontsize=9, color=MUTE)
    ax.set_xticks(X)
    ax.set_xlim(0.7, ND + 0.3)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))
    ax.grid(axis='y', color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(BASELINE)
    ax.tick_params(colors=MUTE, labelsize=8.5)

handles, labels_ = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_, loc='lower center', ncol=3, fontsize=9,
           frameon=False, bbox_to_anchor=(0.5, -0.01), handlelength=2.6)
fig.suptitle('Per-slot draft acceptance by sampling setting\n'
             'conditional on reaching slot j; 40 prompts x B=8 rollouts per prompt per setting',
             fontsize=10.5, color=INK, x=0.01, ha='left', y=1.03)
fig.tight_layout(rect=[0, 0.07, 1, 1])
import os; os.makedirs(DOC + '/figures', exist_ok=True)
fig.savefig(f'{DOC}/figures/slot_acceptance.png', dpi=150, bbox_inches='tight',
            facecolor=SURFACE)
print('saved doc/figures/slot_acceptance.png')

# ---- markdown tables -----------------------------------------------------------
print('\n<!-- tables -->')
for ds in DS_ORDER:
    print(f'\n**{DS_LABELS[ds]}**\n')
    print('| setting | ' + ' | '.join(f'pos {j}' for j in X) + ' |')
    print('|---' * (ND + 1) + '|')
    for m in MODES:
        if m not in slot.get(ds, {}):
            continue
        a = slot[ds][m]['alpha']
        cells = [f'{v*100:.1f}%' if v is not None else '–' for v in a]
        print(f'| {LABELS[m]} | ' + ' | '.join(cells) + ' |')

# save merged slot json for reference
json.dump(slot, open(f'{BASE}/slot_acceptance_all.json', 'w'), indent=2)
print('\nsaved slot_acceptance_all.json')

"""Replot final-paper Figures3/4 from included aggregates; no model execution."""
import argparse
import csv
import json
from pathlib import Path


MODELS = ['ours_gapset_relative_mass_control', 'ours_gapset_state_no_mass_norm_control',
          'ours_gapset_relative_fixed_lag', 'sensefi_resnet18', 'grud_common8', 'hms_rwkv_full']
LABELS = ['GapSet-R', 'No correction', 'Fixed lag', 'ResNet18', 'GRU-D', 'HMS-RWKV']
COLORS = ['#E40000', '#C5AE00', '#51ADE0', '#009E73', '#CD70A8', '#E69500']
MARKERS = ['*', 's', 'o', 'v', '^', 'D']
ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT / 'results')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--font', default='Times New Roman')
    a = p.parse_args()
    if a.output.exists():
        p.error('Choose a new output directory')
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    from matplotlib.lines import Line2D
    from matplotlib.ticker import AutoMinorLocator
    for directory in [Path('C:/Windows/Fonts'), Path('/mnt/c/Windows/Fonts')]:
        for name in ['times.ttf', 'timesbd.ttf', 'timesi.ttf', 'timesbi.ttf']:
            if (directory / name).exists():
                fm.fontManager.addfont(directory / name)
    try:
        fm.findfont(fm.FontProperties(family=a.font), fallback_to_default=False)
        font = a.font
    except ValueError:
        font = 'DejaVu Serif'
    plt.rcParams.update({'font.family': font, 'font.size': 11, 'svg.fonttype': 'none',
                         'pdf.fonttype': 42, 'axes.labelsize': 12,
                         'axes.linewidth': 1.05, 'savefig.facecolor': 'white'})
    a.output.mkdir(parents=True)

    def save(fig, name):
        fig.canvas.draw()
        for suffix in ['png', 'svg']:
            fig.savefig(a.output / f'{name}.{suffix}', dpi=600)
        plt.close(fig)

    with (a.data / 'burst_summary.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(2, 3, figsize=(8.1, 4.85))
    fig.subplots_adjust(left=.07, right=.87, bottom=.08, top=.92, wspace=.30, hspace=.42)
    matrices = []
    for j, (ax, model, label) in enumerate(zip(axes.flat, MODELS, LABELS)):
        own = [r for r in rows if r['model'] == model]
        assert len(own) == 9 and all(int(r['seed_count']) == 3 for r in own)
        values = np.array([[float(next(r['mean_f1'] for r in own
                          if int(r['deleted_percent']) == level and float(r['position']) == pos))
                          for pos in [0., .5, 1.]] for level in [10, 30, 50]])
        matrices.append(values)
        im = ax.imshow(values, cmap='Blues', vmin=0, vmax=80, aspect='auto', interpolation='nearest')
        ax.set_xticks(range(3), ['Start', 'Mid', 'End'])
        ax.set_yticks(range(3), ['10', '30', '50'])
        ax.tick_params(which='both', length=0, pad=5)
        ax.set_title(label, fontsize=12, pad=7)
        if j % 3 == 0:
            ax.set_ylabel('Deletion (%)', labelpad=4)
        ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
        ax.grid(which='minor', color='white', lw=1.5)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i in range(3):
            for k in range(3):
                ax.text(k, i, f'{values[i,k]:.1f}', ha='center', va='center', fontsize=12,
                        color='white' if values[i,k] > 45 else '#152533')
    cb = fig.colorbar(im, cax=fig.add_axes([.90, .08, .014, .84]), ticks=[0, 20, 40, 60, 80])
    cb.set_label('F1 (%)', labelpad=5)
    cb.outline.set_visible(False)
    save(fig, 'figure3_burst_position')

    with (a.data / 'severity.csv').open(encoding='utf-8-sig', newline='') as f:
        raw = list(csv.DictReader(f))
    kinds = ['rf_random', 'frequency_span', 'step_random', 'step_burst']
    series = {}
    for kind in kinds:
        series[kind] = {}
        for model in MODELS:
            own = [r for r in raw if r['model'] == model and r['kind'] == kind]
            xs = sorted({float(r['deleted_percent']) for r in own})
            values = [[float(r['macro_f1']) for r in own if float(r['deleted_percent']) == x] for x in xs]
            assert len(xs) == 4 and all(len(v) == 3 for v in values)
            series[kind][model] = {'x': xs, 'mean': np.mean(values, axis=1).tolist(),
                                   'sample_sd': np.std(values, axis=1, ddof=1).tolist()}
    reference = json.loads((a.data / 'severity_summary.json').read_text(encoding='utf8'))
    for kind in kinds:
        for model in MODELS:
            for key in ['x', 'mean', 'sample_sd']:
                np.testing.assert_allclose(series[kind][model][key], reference[kind][model][key], atol=1e-10)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.9))
    fig.subplots_adjust(left=.085, right=.985, bottom=.09, top=.855, wspace=.29, hspace=.39)
    for ax, kind in zip(axes.flat, kinds):
        for j in [1, 2, 3, 4, 5, 0]:
            s = series[kind][MODELS[j]]
            x, y, sd = np.array(s['x']), np.array(s['mean']), np.array(s['sample_sd'])
            ax.fill_between(x, y-sd, y+sd, color=COLORS[j], alpha=.085, linewidth=0, zorder=1)
            ax.plot(x, y, color=COLORS[j], marker=MARKERS[j], ms=8 if j == 0 else 5,
                    mew=.45, lw=1.25 if j == 0 else 1.1, zorder=4 if j == 0 else 3)
        xs = series[kind][MODELS[0]]['x']
        ax.set(xlim=(-2, max(xs)+2), ylim=(0, 85), xticks=xs, yticks=[0, 20, 40, 60, 80],
               xlabel='Deletion (%)', ylabel='Macro-F1 (%)')
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(which='major', direction='out', length=3.5, width=.85, pad=3)
        ax.tick_params(which='minor', direction='out', length=2, width=.55)
        ax.set_axisbelow(True)
        ax.grid(which='major', axis='y', color='#939393', lw=.75)
        ax.grid(which='major', axis='x', color='#C8C8C8', lw=.55)
        ax.grid(which='minor', color='#DEDEDE', lw=.45, linestyle=':')
    handles = [Line2D([], [], color=c, marker=m, linewidth=1.2, markersize=8 if m == '*' else 5,
                     markeredgewidth=.45) for c, m in zip(COLORS, MARKERS)]
    fig.legend(handles, LABELS, ncol=3, loc='upper center', bbox_to_anchor=(.54, .995),
               frameon=False, handlelength=2.15, handletextpad=.45, columnspacing=1.6, labelspacing=.5)
    save(fig, 'figure4_missingness')
    (a.output / 'plot_values.json').write_text(json.dumps({'severity': series, 'font': font,
         'burst': {m: v.tolist() for m, v in zip(MODELS, matrices)},
         'scope': 'existing three-seed development aggregates; no new scoring',
         'frequency_span': 'contiguous subcarriers retained; x is deleted proportion',
         'burst_sd': 'sample SD available in results/burst_summary.csv'}, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()

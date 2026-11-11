from io import BytesIO
import matplotlib.pyplot as plt


def render_report(frame, company, frequency, unit, groups, colors):
    fig, axes = plt.subplots(len(groups), 1, figsize=(13, 4.5 * len(groups)), constrained_layout=True)
    if len(groups) == 1:
        axes = [axes]
    for ax, (title, names) in zip(axes, groups):
        values = frame[names]
        x = range(len(values.index))
        width = 0.8 / max(len(names), 1)
        for idx, name in enumerate(names):
            ax.bar([i + (idx-(len(names)-1)/2)*width for i in x], values[name].to_numpy(), width=width, label=str(name))
        ax.set_title(str(title)); ax.set_xticks(list(x), [str(v) for v in values.index]); ax.set_ylabel(unit); ax.grid(axis='y', alpha=.2); ax.legend(fontsize=8, ncol=min(3,len(names)))
    fig.suptitle(f"{company} Financial Dashboard · {frequency}", fontsize=18)
    out=BytesIO(); fig.savefig(out, format='png', dpi=160, bbox_inches='tight'); plt.close(fig); return out.getvalue()

"""Script that display a pair plot matrix
of a dataset given at run time. The data must be a .csv file."""


import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from utils.load_data import load_data
from utils.houses import get_houses


def disp_pair_plot(df: pd.DataFrame):
    """Create and display pair plot."""
    if 'Index' in df.columns:
        df = df.drop('Index', axis=1)

    df_plot = df.rename(columns={
        'Defense Against the Dark Arts': 'Defense',
        'Care of Magical Creatures': 'Care of Creatures',
        'History of Magic': 'Hist. Magic',
        'Ancient Runes': 'Runes'
    })

    sns.set_theme(style="ticks")
    fig = sns.pairplot(
        df_plot,
        hue="Hogwarts House",
        diag_kind="kde",
        palette="Set1",
        height=1.4,
        plot_kws={"alpha": 0.5},
        corner=True,
    )

    for ax in fig.axes.flatten():
        if ax is None:
            continue
        ax.tick_params(labelsize=6)
        ax.set_xlabel(
            ax.get_xlabel(),
            fontsize=8,
            rotation=30,
            ha='right',
            fontstyle='italic'
        )
        ax.set_ylabel(
            ax.get_ylabel(),
            fontsize=8,
            rotation=0,
            ha='right',
            va='center',
            fontstyle='italic'
        )

    fig.figure.suptitle("Pair Plot - Hogwarts Houses",
                        fontsize=16, fontweight='bold')
    fig.figure.subplots_adjust(top=0.95, left=0.08,
                               bottom=0.08, hspace=0.15, wspace=0.15)

    out_dir = "plots"
    try:
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, "pair_plot.png"), dpi=150)
    except PermissionError:
        print(f"Error: no write permission on '{out_dir}'.",
              file=sys.stderr)
    except OSError as error:
        print(f"Error: could not save figure: {error}.",
              file=sys.stderr)

    try:
        plt.show()
    except KeyboardInterrupt:
        print("\nPair plot closed.")
        sys.exit(130)


def main():
    """Run the script and get the file as parameter."""

    ac = len(sys.argv)
    if ac != 2:
        print("Error: the program must take one .csv file as argument.",
              file=sys.stderr)
        sys.exit(1)

    df = load_data(sys.argv[1])
    get_houses(df)
    disp_pair_plot(df)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPair plot closed.")
        sys.exit(130)

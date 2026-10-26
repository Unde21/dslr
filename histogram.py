"""Script that display an histogram of dataset given at run time.
The data must be a .cvs file."""


import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from utils.load_data import load_data
from utils.houses import get_houses


def disp_histogram(df: pd.DataFrame):
    """Create and display all histogram, to answer the question:
        Which Hogwarts course has homogeneous score distribution
        between all four houses?"""
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if 'Index' in numeric_cols:
        numeric_cols = numeric_cols.drop('Index')

    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(20, 20))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        ax = axes[i]

        sns.histplot(
            data=df,
            x=col,
            hue="Hogwarts House",
            element="step",
            stat="density",
            common_norm=False,
            ax=ax,
            palette="Set1"
        )

        ax.set_title(col, fontsize=15, fontweight='bold')
        ax.set_xlabel(col, fontsize=12, fontstyle='italic')
        ax.set_ylabel("Density", fontsize=12, fontstyle='italic')
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.subplots_adjust(top=0.88, bottom=0.12, hspace=0.6, wspace=0.5)
    fig.suptitle("Distribution of matter for each House of Hogwarts",
                 fontsize=25, fontweight='bold', y=0.95)

    houses = sorted(df["Hogwarts House"].dropna().unique())
    if not houses:
        print("Error: column 'Hogwarts House' contains no house.",
              file=sys.stderr)
        sys.exit(1)
    colors = sns.color_palette("Set1", len(houses))
    patches = [mpatches.Patch(color=colors[i],
                              label=houses[i]) for i in range(len(houses))]

    fig.legend(handles=patches, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, 0.02), fontsize=12)

    out_dir = "plots"
    try:
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, "histogram.png"), dpi=150)
    except PermissionError:
        print(f"Error: no write permission on '{out_dir}'.",
              file=sys.stderr)
    except OSError as error:
        print(f"Error: could not save figure: {error}.",
              file=sys.stderr)

    try:
        plt.show()
    except KeyboardInterrupt:
        print("\nHistogram display closed.")
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
    disp_histogram(df)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nHistogram display closed.")
        sys.exit(130)

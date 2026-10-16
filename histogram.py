"""Script that display an histogram of dataset given at run time.
The data must be a .cvs file."""


import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

def load_csv(path: str) -> pd.DataFrame:
    """Function that receive Dataset in csv format,
    and return it as Pandas DataFrame if none error occured."""

    try:
        if not isinstance(path, str):
            raise TypeError("path must be a string.")
        if not path.lower().endswith((".csv")):
            raise ValueError("file extension must be .csv")

        df = pd.read_csv(path)

        if df.empty:
            raise ValueError("csv file is empty.")

        return df

    except (TypeError, ValueError, KeyError,
            FileNotFoundError, PermissionError,
            IsADirectoryError) as error:
        print(f"{type(error).__name__}: {error}")
        sys.exit(1)


def disp_histogram(df: pd.DataFrame):
    """Create and display all histogram."""
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
    colors = sns.color_palette("Set1", len(houses))
    patches = [mpatches.Patch(color=colors[i], label=houses[i]) for i in range(len(houses))]
    
    fig.legend(handles=patches, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, 0.02), fontsize=12)
    try:
        plt.show()
    except KeyboardInterrupt:
        print("Histogram display canceled.")
        sys.exit(0)


def main():
    """Run the script and get the file as param"""

    ac = len(sys.argv)

    try:
        assert ac == 2, "the program must take one .csv file as param."
    except AssertionError as error:
        print(f"{type(error).__name__}: {error}")
        sys.exit(1)

    df = load_csv(sys.argv[1])
    disp_histogram(df)
    print(df)


if __name__ == "__main__":
    main()


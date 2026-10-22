"""Script that display an scatter plot of dataset given at run time.
The data must be a .csv file."""

import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
from utils.load_data import load_data


def disp_scalter_plot(df: pd.DataFrame):
    """Create and display scalter plot, to answer the question:
    What are the two features that are similar?"""
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    if 'Index' in numeric_df:
        numeric_df = numeric_df.drop('Index', axis=1)

    corr_matrix = numeric_df.corr()
    corr_df = corr_matrix.stack().reset_index()
    corr_df.columns = ['Feature_1', 'Feature_2', 'Correlation']
    corr_df = corr_df[corr_df['Feature_1'] < corr_df['Feature_2']]
    corr_df['Abs_Correlation'] = corr_df['Correlation'].abs()
    sorted_corr = corr_df.sort_values(by='Abs_Correlation', ascending=False)

    best_match = sorted_corr.iloc[0]
    feature1 = best_match['Feature_1']
    feature2 = best_match['Feature_2']
    score = best_match['Correlation']

    print("Feature that are the most similar are :")
    print(f" -> {feature1} and {feature2} (Correlation: {score:.4f})")

    df_clean = df.dropna(subset=[feature1, feature2])

    plt.figure(figsize=(16, 15))

    if 'Hogwarts House' in df.columns:
        houses = df_clean['Hogwarts House'].unique()
        colors = {
            'Gryffindor': 'red',
            'Slytherin': 'green',
            'Ravenclaw': 'blue',
            'Hufflepuff': 'yellow'
        }

        for h in houses:
            house_data = df_clean[df_clean['Hogwarts House'] == h]
            plt.scatter(
                house_data[feature1],
                house_data[feature2],
                alpha=0.6,
                label=h,
                color=colors.get(h, 'gray')
            )
        plt.legend()

    else:
        plt.scatter(df_clean[feature1], df_clean[feature2],
                    alpha=0.6, color='purple')

    plt.title(f"Scatter Plot: {feature1} vs {feature2}",
              fontsize=25, fontweight='bold')
    plt.xlabel(feature1, fontsize=15, fontstyle='italic')
    plt.ylabel(feature2, fontsize=12, fontstyle='italic')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    out_dir = "plots"
    try:
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, "scatter_plot.png"), dpi=150)
    except PermissionError:
        print(f"Error: no write permission on '{out_dir}'.",
              file=sys.stderr)
    except OSError as error:
        print(f"Error: could not save figure: {error}.",
              file=sys.stderr)

    try:
        plt.show()
    except KeyboardInterrupt:
        print("\nScatter Plot closed.")
        sys.exit(0)


def main():
    """Run the script and get the file as parameter."""

    ac = len(sys.argv)
    if ac != 2:
        print("Error: the program must take one .csv file as argument.",
              file=sys.stderr)
        sys.exit(1)

    df = load_data(sys.argv[1])
    disp_scalter_plot(df)


if __name__ == "__main__":
    main()

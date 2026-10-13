"""Main function that run the program."""

from load_data import load_data
from describe import ft_describe


def main():
    """Main function"""

    df = load_data("datasets/dataset_train.csv")

    print(df)

    print("\ndescribe")
    print("=" * 45)
    print(df.describe())

    print("\nMy describe")
    print("=" * 45)
    print(ft_describe(df))


if __name__ == "__main__":
    main()

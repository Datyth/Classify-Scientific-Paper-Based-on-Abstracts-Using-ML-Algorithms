import pandas as pd


if __name__ == "__main__":
    df = pd.read_csv('train_split.csv')
    print(len(df))
    print(df[0:5])
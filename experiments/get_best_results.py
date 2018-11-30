import pandas as pd
import numpy as np
import sys

pd.options.display.max_colwidth = 100


def get_scores(filename):
    df = pd.read_csv(filename, index_col=0)
    df = df.iloc[:2]  # take first result
    df = df.stack()
    df['filename'] = filename  # add filename to the series
    return df


if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.exit('Give a list of filenames to csv result files')
    filenames = sys.argv[1:]
    # combine csv files
    df = pd.DataFrame([get_scores(f) for f in filenames])
    df.sort_values(by=[('training', 'logp')], inplace=True)
    print('Best 5 results:')
    print(df.tail(5))
    print('Summary:')
    print(df.tail(5).describe())

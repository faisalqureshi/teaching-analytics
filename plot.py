import pandas as pd
import argparse
import pprint
import numpy as np

global ignore_columns

def read_file(filename, sheet_name):
    try:
        df = pd.read_excel(args.filename, sheet_name=sheet_name)
        return df
    except:
        print('Error reading {}.  Nothing to do here.'.format(args.filename))
    return pd.DataFrame()

def inspect_columns(df):
    i = 0
    for name in df.columns.values:
        print('\t{:<10}{:40.40}{:16.16}'.format(i,name,str(df[name].dtype)))
        i = i+1

def compute_stats(df):
    stats = []
    for name in df.columns.values:
        if name in ignore_columns:
            print('Ignoreing {}.  Found in ignore list.'.format(name))
        elif not df[name].dtype in ['float64', 'float32', 'int', 'int32']:
            print('Ignoreing {}.  Non-numeric datatype.'.format(name))
        else:
            print('Processing {}'.format(name))
            data = np.nan_to_num(df[name])
            positives = data > 0
            column_stats = {'name': name,  'mean': np.mean(data), 'median': np.median(data), 'max': np.max(data), 'min': np.min(data[positives])}
            stats.append(column_stats)
    return stats

def print_stats(stats):
    print('\n')
    print('{:>40.40}{:>10.10}{:>10.10}{:>10.10}{:>10.10}'.format('Name','Mean','Median','Min','Max'))
    for s in stats:
        print('{name:>40.40}{mean:10.2f}{median:10.2f}{min:10.2f}{max:10.2f}'.format(**s))

if __name__ == '__main__':

    ignore_columns = [
        'Last Name',
        'First Name',
        'Username',
        'Student ID',
        'Last Access',
        'Availability',
        'Child Course ID'
    ]

    parser = argparse.ArgumentParser(description='Students score analysis.')
    parser.add_argument('filename', help='Excel containing student scores.')
    parser.add_argument('-i','--inspect', action='store_true', default='False', help='Prints out column headers.')
    parser.add_argument('-c','--col', action='store', default='', help='Specify columns that will be considered for analysis.')
    # parser.add_argument('-a','--action', action='store', default='hist', help='Specify the plot that needs to be generated.  Default is "hist".')
    # parser.add_argument('-s','--sheet', action='store', default=None, help='Specify the sheet to use for data reading.')
    parser.add_argument('--stats', action='store_true', default=False, help='Compute first order statistics for different columns.')
    parser.add_argument('--show-ignore-columns', action='store_true', default=False, help='Displays the list of ignored columns.')

    args = parser.parse_args()
    print(args)

    if args.show_ignore_columns:
        print('The following columns are always ignored.')
        for name in ignore_columns:
            print('\t', name)
        exit(0)

    if not args.sheet: sheet_name = 0

    df = read_file(args.filename, sheet_name)
    if df.empty:
        exit(-1)

    if args.inspect == True:
        inspect_columns(df)
        exit(0)

    if args.stats == True:
        stats = compute_stats(df)
        print_stats(stats)
        exit(0)
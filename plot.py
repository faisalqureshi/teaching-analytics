import pandas as pd
import argparse
import pprint
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

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
    print('\n\t{:<10}{:40.40}{:16.16}'.format('Index','Name','Data type'))
    for name in df.columns.values:
        print('\t{:<10}{:40.40}{:16.16}'.format(i,name,str(df[name].dtype)))
        i = i+1

def compute_stats(df, include_columns):
    stats = []
    for name in include_columns:
        print('Processing {}'.format(name))
        data = np.nan_to_num(df[name])
        positives = data > 0
        d = data[positives]
        column_stats = {'name': name,  'mean': np.mean(d), 'median': np.median(d), 'max': np.max(d), 'min': np.min(d)}
        stats.append(column_stats)
    return stats

def print_stats(stats):
    print('\n')
    print('{:>40.40}{:>10.10}{:>10.10}{:>10.10}{:>10.10}'.format('Name','Mean','Median','Min','Max'))
    for s in stats:
        print('{name:>40.40}{mean:10.2f}{median:10.2f}{min:10.2f}{max:10.2f}'.format(**s))

def get_include_columns(df, cols, totals):
    if not cols:
        indices = np.arange(len(df.columns.values))
    else:
        indices = list(map(int,cols.split(',')))

    if not totals:
        totals = np.ones(len(indices))*100
    else:
        totals = list(map(int,totals.split(',')))

    if len(indices) != len(totals):
        print('Totals provided do not match column indices')
        exit(-3)

    include_columns = []
    include_columns_totals = []
    i = 0
    for name in df.columns.values:
        if i in indices:
            if name not in ignore_columns and df[name].dtype == 'float64':
                include_columns.append(name)
                include_columns_totals.append(totals[indices.index(i)])
            else:
                print('Cannot add {}.'.format(name))
        i = i+1
    return include_columns, include_columns_totals

def plot_corr(df, include_columns, include_columns_totals):

    name, total = include_columns[0], include_columns_totals[0]
    x = np.nan_to_num(df[name]) * 100 / total

    name, total = include_columns[1], include_columns_totals[1]
    y = np.nan_to_num(df[name]) * 100 / total

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis('equal')
    ax.set_title('Corr. {} vs. {}'.format(include_columns[0],include_columns[1]))
    ax.scatter(x, y, alpha=0.5)
    ax.set_xlim(0,100)
    ax.set_ylim(0,100)
    ax.set_xlabel('% Marks')
    ax.set_ylabel('% Marks')
    plt.show()

def plot_hist(df, include_columns, include_columns_totals):
    i = 0
    
    for name in include_columns:
        data = np.nan_to_num(df[name]) * 100 / include_columns_totals[i]
        positives = data > 0
        d = data[positives]

        fig = plt.figure()
        #ax = fig.add_subplot(len(include_columns),1,i+1)
        ax = fig.add_subplot(111)
        ax.text(0.1,0.9,'Mean={:6.2f}'.format(np.mean(d)), transform=ax.transAxes, color='black')
        ax.text(0.1,0.85,'Median={:6.2f}'.format(np.median(d)), transform=ax.transAxes, color='black')
        ax.text(0.1,0.8,'Max={:6.2f}'.format(np.max(d)), transform=ax.transAxes, color='black')
        ax.text(0.1,0.75,'Min={:6.2f}'.format(np.min(d)), transform=ax.transAxes, color='black')
        ax.set_title(name)
        ax.hist(data,bins=20)
        ax.set_xlabel('% Marks')
        ax.set_xlim(0,100)
        i += 1
    plt.show()

if __name__ == '__main__':

    ignore_columns = [
        'Username',
        'Student ID',
        'Child Course ID'
    ]

    parser = argparse.ArgumentParser(description='Students score analysis.')
    parser.add_argument('filename', help='Excel containing student scores.')
    parser.add_argument('-i','--inspect', action='store_true', default='False', help='Prints out column headers.')
    parser.add_argument('-c','--columns', action='store', default=None, help='Specify columns that will be considered for analysis.')
    parser.add_argument('-t','--totals', action='store', default=None, help='Specify total marks for provided columns.')
    parser.add_argument('-a','--action', action='store', default='hist', help='Specify the plot that needs to be generated.  Default is "hist".')
    # parser.add_argument('-s','--sheet', action='store', default=None, help='Specify the sheet to use for data reading.')
    parser.add_argument('--stats', action='store_true', default=False, help='Compute first order statistics for different columns.')

    args = parser.parse_args()
    print(args)

    # if not args.sheet: sheet_name = 0
    sheet_name = 0

    df = read_file(args.filename, sheet_name)
    if df.empty:
        exit(-1)

    if args.inspect == True:
        inspect_columns(df)
        exit(0)

    include_columns, include_columns_totals = get_include_columns(df, args.columns, args.totals)
    if len(include_columns) == 0:
        print('No columns found to analyze.  Nothing to do here.')
        exit(-2)

    if args.stats == True:
        stats = compute_stats(df, include_columns)
        print_stats(stats)
        exit(1)

    if args.action == 'hist':
        plot_hist(df, include_columns, include_columns_totals)
    elif args.action == 'corr':
        if len(include_columns) > 2:
            print('Please specify only two columns to plot correlatons.')
            exit(-4)
        plot_corr(df, include_columns, include_columns_totals)
    else:
        print('Unknown action {}.  Nothing to do here.'.format(args.action))
        exit(-5)
    exit(2)

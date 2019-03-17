import pandas as pd
import argparse
import pprint
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

ignore_columns = [
    'Username',
    'Student ID',
    'Child Course ID'
]

grades_ranges = [
    0,
    50,
    60,
    67,
    70,
    73,
    77,
    80,
    85,
    90,
    100
]

letter_grades = [
    'F',
    'D',
    'C',
    'C+',
    'B-',
    'B',
    'B+',
    'A-',
    'A',
    'A+'
]

def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])

    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

def read_file(filename, sheet_name=0):
    try:
        if filename.endswith(('.xlsx', '.xls', '.xlsm')):
            df = pd.read_excel(filename, sheet_name=sheet_name)
        else:
            df = pd.read_csv(filename)
        return df
    except:
        print('Error reading {}.  Nothing to do here.'.format(filename))
    return pd.DataFrame()

def inspect_columns(df):
    i = 0
    print('\n\t{:<10}{:40.40}{:16.16}'.format('Index','Name','Data type'))
    for name in df.columns.values:
        print('\t{:<10}{:40.40}{:16.16}'.format(i,name,str(df[name].dtype)))
        i += 1

def compute_stats(df, include_columns, include_columns_totals):
    stats = []
    
    i = 0
    for name in include_columns:
        print('Processing {}'.format(name))
        data = np.nan_to_num(df[name].astype('float32')) * 100 / include_columns_totals[i]
        column_stats = {'name': name,  'mean': np.mean(data), 'mean2': np.mean(data[data>0]), 'median': np.median(data), 'median2': np.median(data[data>0]), 'max': np.max(data), 'min': np.min(data[data>0])}
        stats.append(column_stats)
        i = i+1
    return stats

def print_stats(stats, title=''):
    print('\n{:^100.100}'.format(title))
    print('{:>40.40}{:>10.10}{:>10.10}{:>10.10}{:>10.10}{:>10.10}{:>10.10}'.format('Name','Mean0','Mean','Median0','Median','Min','Max'))
    for s in stats:
        print('{name:>40.40}{mean:10.2f}{mean2:10.2f}{median2:10.2f}{median:10.2f}{min:10.2f}{max:10.2f}'.format(**s))

def get_include_columns(df, cols, totals):
    if not cols:
        indices = np.arange(len(df.columns.values)).tolist()
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
            if name not in ignore_columns and df[name].dtype in ['float64','int64']:
                include_columns.append(name)
                include_columns_totals.append(totals[indices.index(i)])
            else:
                print('Cannot add {}.'.format(name))
        i = i+1
    return include_columns, include_columns_totals

def plot_corr(df, include_columns, include_columns_totals, title=''):
    i = 0
    name, total = include_columns[i], include_columns_totals[i]
    x = np.nan_to_num(df[name].astype('float32')) * 100 / total

    i = 1
    name, total = include_columns[i], include_columns_totals[i]
    y = np.nan_to_num(df[name].astype('float32')) * 100 / total

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis('equal')
    ax.set_title(title,fontsize=18)
    ax.scatter(x, y, alpha=0.5)
    ax.text(0.1,0.91,'Corr. {} vs. {}'.format(include_columns[0],include_columns[1]),transform=ax.transAxes, color='black',fontsize=14)
    ax.set_xlim(0,100)
    ax.set_ylim(0,100)
    ax.set_xlabel('% Marks')
    ax.set_ylabel('% Marks')
    plt.show()

def plot_grades(df, include_columns, include_columns_totals, title='', ranges=None):
    if not ranges:
        ranges = grades_ranges
    else:
        ranges = np.array(list(map(int,ranges.split(','))))
    
    i = 0
    for name in include_columns:
        print('\n\t{:40.40}'.format(name))
        data = np.nan_to_num(df[name].astype('float32')) * 100 / include_columns_totals[i]
        counts, edges = np.histogram(data, ranges)
        sum = np.sum(counts)
        
        print('\t{:10}{:>10}{:>20.20}'.format('Letter', 'Count', 'Percentage'))
        j = 0
        for j in range(len(counts)):
            print('\t{:10}{:>10}{:>20.2f}'.format(letter_grades[j], str(counts[j]), counts[j]*100/sum ))
            j = j + 1
        print('\t{:15.15}{:>5}'.format('Total students',str(len(data))))
        if sum != len(data):
            print('Warning: sum of counts is not equal to the number of data items.  Check ranges.')
            print('\t{:15.15}{:>5}'.format('Count',str(sum)))
        
        c = mcolors.ColorConverter().to_rgb
        rvb = make_colormap([c('red'), 0.125, c('red'), c('orange'), 0.25, c('orange'),c('green'),0.5, c('green'),0.7, c('green'), c('blue'), 0.75, c('blue')])
        fig = plt.figure()
        fig.suptitle(title, fontsize=18)
        ax = fig.add_subplot(111)
        n = len(counts)
        x = np.arange(len(counts))
        if len(counts) == len(letter_grades):
            ax.bar(x, counts*100/sum, tick_label=letter_grades, alpha=0.95, color=rvb(x/n))
        else:
            ax.bar(x, counts*100/sum, alpha=0.95, color=rvb(x/n))
        ax.set_xlabel('Grades')
        ax.set_ylabel('Percentages')
        ax.text(0.02,0.91,name,fontsize=16,transform=ax.transAxes)
        ax.text(0.02,0.85,'Total students {}'.format(str(len(data))),transform=ax.transAxes)
        if sum != len(data):
            ax.text(0.02,0.75,'Count {}'.format(sum),transform=ax.transAxes, color='red')
            ax.text(0.02,0.8,'Warning: check ranges', transform=ax.transAxes, color='red')
        i += 1
    plt.show()

def plot_hist(df, include_columns, include_columns_totals, title=''):
    ranges = np.arange(101)
    
    i = 0
    for name in include_columns:
        data = np.nan_to_num(df[name]) * 100 / include_columns_totals[i]
        counts, edges = np.histogram(data, bins=ranges)
        print(counts)
        centers = 0.5*(edges[1:]+ edges[:-1])
        sum = np.sum(counts)
        if sum != len(data):
            print('Warning: sum of counts is not equal to the number of data items.  Check ranges.')
            print('\t{:15.15}{:>5}'.format('Count',str(sum)))

        c = mcolors.ColorConverter().to_rgb
        rvb = make_colormap([c('red'), 0.125, c('red'), c('orange'), 0.25, c('orange'),c('green'),0.5, c('green'),0.7, c('green'), c('blue'), 0.75, c('blue')])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.text(0.02,0.85,'Mean={:6.2f},{:6.2f}'.format(np.mean(data), np.mean(data[data>0])), transform=ax.transAxes, color='black')
        ax.text(0.02,0.8,'Median={:6.2f},{:6.2f}'.format(np.median(data), np.median(data[data>0])), transform=ax.transAxes, color='black')
        ax.text(0.02,0.75,'Max={:6.2f}'.format(np.max(data)), transform=ax.transAxes, color='black')
        ax.text(0.02,0.7,'Min={:6.2f}'.format(np.min(data[data > 0])), transform=ax.transAxes, color='black')
        ax.text(0.02,0.91,name,transform=ax.transAxes, color='black',fontsize=14)
        ax.text(0.02,0.65,'Total students={:6.6}'.format(str(len(counts))), transform=ax.transAxes, color='black')
        ax.set_title(title,fontsize=18)
        if sum != len(data):
            ax.text(0.02,.6,'Warning: check ranges.', color='red', transform=ax.transAxes)
            ax.text(0.02,.55,'{:15.15}{:>5}'.format('Count',str(sum)), color='red', transform=ax.transAxes)
        x = np.arange(len(counts))
        n = len(counts)
        ax.bar(centers, counts, alpha=0.95, color=rvb(x/n), width=1)
        ax.set_xlabel('% Marks')
        ax.set_xlim(0,100)
        i += 1
    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Students score analysis.')
    parser.add_argument('filename', help='Excel containing student scores.')
    parser.add_argument('-i','--inspect', action='store_true', default='False', help='Prints out column headers.  Super useful to identify columns by their indices.')
    parser.add_argument('-c','--columns', action='store', default=None, help='Specify columns that will be considered for analysis.  To analyze columns 4, 6, and 8, simply say --columns=4,6,8.')
    parser.add_argument('-r','--ranges', action='store', default=None, help='Specify ranges for grade historgam.  Default values are 0,50,60,67,70,73,77,80,85,90,100.')
    parser.add_argument('-t','--totals', action='store', default=None, help='Specify total marks for provided columns.  This is needed to give percentage scores.')
    parser.add_argument('-a','--action', action='store', default='hist', help='Specify the plot that needs to be generated.  Default is "hist".  Supported actions are "hist", "corr", and "grades".  "corr" plots a scatter plot capturing teh correlation between two columns.')
    # parser.add_argument('-s','--sheet', action='store', default=None, help='Specify the sheet to use for data reading.')
    parser.add_argument('--stats', action='store_true', default=False, help='Compute first order statistics for different columns.')
    parser.add_argument('--title', action='store', default='', help='Provide a title for the plots.')

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
        stats = compute_stats(df, include_columns, include_columns_totals)
        print_stats(stats, args.title)
        exit(1)

    if args.action == 'hist':
        plot_hist(df, include_columns, include_columns_totals, title=args.title)
    elif args.action == 'corr':
        if len(include_columns) != 2:
            print('Please specify only two columns to plot correlatons.')
            exit(-4)
        plot_corr(df, include_columns, include_columns_totals, title=args.title)
    elif args.action == 'grades':
        plot_grades(df, include_columns, include_columns_totals, ranges=args.ranges, title=args.title)
    else:
        print('Unknown action {}.  Nothing to do here.'.format(args.action))
        exit(-5)
    exit(2)

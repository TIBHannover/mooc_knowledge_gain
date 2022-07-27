import argparse
import os
import pandas as pd
import glob
import re
import numpy as np
import matplotlib.pyplot as plt

from os.path import basename


required_columns = ['Key_Dataset', 'Percent_correct', 'Kappa_statistic', 'Mean_absolute_error', 'Root_mean_squared_error',
                    'Relative_absolute_error', 'Root_relative_squared_error', 'IR_precision', 'IR_recall', 'F_measure', 
                    'Kappa_statistic', 'Matthews_correlation']


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=True, help='the path of the weka csvs')
    parser.add_argument('-k', '--k_fold', type=int, required=True, help='used k_fold for calculation')
    args = parser.parse_args()

    return args


def preprocess_csv(csv_path):
    ''' Removes unwanted columns from a csv-object
    :param csv_path: path of csv
    :return: edited csv
    '''
    df = pd.read_csv(csv_path, encoding='utf-8')
    # remove unwanted columns
    df = df[required_columns]
    #print(df)
    return df


def calculate_class_average(df, k):
    ''' Calculates the average of the k-fold results of a csv-file
    :param df: loaded csv-file
    :param k: k parameter of k-fold
    :return: averaged values as csv-object
    '''

    # get length of different feature classes
    file_names = df.iloc[:, 0].tolist()
    all_len = sum('all_features' in s for s in file_names)
    embd_len = sum('embedding' in s for s in file_names)
    multi_len = sum('multimedia_features' in s for s in file_names)
    text_len = sum('text_features' in s for s in file_names)

    splits = df.copy()

    # create new index array
    indexes = np.concatenate([[i % int(all_len / k) for i in range(all_len)], 
                        [i % int(embd_len / k) + int(all_len / k) for i in range(embd_len)],
                        [(i % int(multi_len / k)) + int((all_len + embd_len) / k)  for i in range(multi_len)],
                        [(i % int(text_len / k)) + int((all_len + embd_len + multi_len) /  k) for i in range(text_len)]], axis=None)

    splits.index = indexes
    changed = pd.MultiIndex.from_frame(splits)
    # caluclate average for every subset
    groups = changed.groupby(splits.index)
    pattern = r'_\d+_'
    averaged = []
    std = []
    for i in range(len(groups)):
        group = groups[i]
        name = re.sub(pattern, '_', group[0][0])
        values = np.zeros((len(required_columns) - 1), dtype=object)
        # calculate standard derivation
        std.append([name] + groups[i].to_frame().std().to_numpy().tolist())
        # calculate average
        for current in group:
            for i in range(len(required_columns) - 1):
                values[i] += current[i+1]
        # average the values
        values = values / k
        # create name for average
        values = np.insert(values, 0, name)
        averaged.append(values.tolist())
    return averaged, std


def calculate_predictor_average(low, moderate, high):
    averaged = []
    std_all = []

    for i in range(len(low)):
        # creaty zero array for averaging (low, moderate, high have the same length)
        values = np.zeros((len(required_columns) - 1), dtype=object)
        # calculate average and standard derivation
        std = [low[i][0]]
        for j in range(len(required_columns) - 1):
            values[j] += low[i][j+1] + moderate[i][j+1] + high[i][j+1]
            std.append(np.std([low[i][j+1], moderate[i][j+1], high[i][j+1]]))

        values = np.insert(values / 3, 0, low[i][0])
        averaged.append(values.tolist())
        std_all.append(std)
    #print(averaged)
    return averaged, std_all


def visualize_results(averaged, fn):
    # get length of different feature classes
    file_names = [x[0] for x in averaged]
    all_len = sum('all_features' in s for s in file_names)
    embd_len = sum('embedding' in s for s in file_names)
    multi_len = sum('multimedia_features' in s for s in file_names)
    #text_len = sum('text_features' in s for s in file_names)

    # store visualization for all features
    a = averaged[:all_len]
    prec_all = [round(item[1], 2) for item in a]
    f1_all = [round(item[3] * 100, 2) for item in a]
    x_all = [get_feature_amount(item[0]) for item in a]
    visualize_result(x_all, prec_all, f1_all, fn + '_all_features')
    # store visulaization for embedding features
    embd = averaged[all_len:all_len + embd_len]
    prec_embd = [round(item[1], 2) for item in embd]
    f1_embd = [round(item[3] * 100, 2) for item in embd]
    x_emd = [get_feature_amount(item[0]) for item in embd]
    visualize_result(x_emd, prec_embd, f1_embd, fn + '_all_features')    
    # store visualization for multimedia features
    multi = averaged[all_len+embd_len:all_len + embd_len + multi_len]
    prec_multi = [round(item[1], 2) for item in multi]
    f1_multi = [round(item[3] * 100, 2) for item in multi]
    x_multi = [get_feature_amount(item[0]) for item in multi]
    visualize_result(x_multi, prec_multi, f1_multi, fn + '_multimedia_features')
    # store visualization for text features
    text = averaged[multi_len+ + embd_len + all_len:]
    prec_text = [round(item[1], 2) for item in text]
    f1_text = [round(item[3] * 100, 2) for item in text]
    x_text = [get_feature_amount(item[0]) for item in text]
    visualize_result(x_text,prec_text, f1_text, fn + '_text_features')


def get_feature_amount(name):
    pattern = r'_\d+features_'
    matches = re.search(pattern, name)

    # check if object is embedding features
    if not matches:
        pattern = r'only|both'
        matches = re.search(pattern, name)
        if matches[0] == 'both':
            return 2
        return 1

    # extract feature amount and return it
    return int(matches[0][:-len('features_')][1:])


def visualize_result(x, precision, f1, name):
    # get correct order for drawing
    x, precision, f1 = zip(*sorted(zip(x,precision, f1),key=lambda x: x))
    # visualize precision
    plt.plot(x, precision, '-o')
    plt.title(name)
    plt.xlabel('amount of features')
    plt.ylabel('avg precision in %')
    plt.xlim(0, np.amax(x) + 5)
    plt.ylim(0, 100)
    plt.savefig('./generated_results/' + name + '_precision.png')
    plt.clf()

    # visualize F1-Score
    plt.plot(x, f1, '-o')
    plt.title(name)
    plt.xlabel('amount of features')
    plt.ylabel('avg f1-score in %')
    plt.xlim(0, np.amax(x) + 5)
    plt.ylim(0, 100)
    plt.savefig('./generated_results/' + name + '_f1.png')
    plt.clf()


def create_csv(low, std_low, moderate, std_moderate, high, std_high, averaged, std_averaged, fn):
    class_names = ['', '', '', '', 'Low', '', '', '', 'Moderate', '', '', '', 'High', '', '', '', 'Average']
    value_names = ['Dataset', 'P', 'std P', 'R', 'std R', 'F1', 'std F1', 'P', 'std P', 'R', 'std R', 'F1', 'std F1',
                   'P', 'std P', 'R', 'std R', 'F1', 'std F1', 'P', 'std P', 'R', 'std R', 'F1', 'std F1',
                   'Percent_correct', 'std Percent_correct']
    csv = [class_names, value_names]
    for i in range(len(low)):
        # the standard derivation for the overall precision can be used from low, moderate or high (same values)
        row = np.round([low[i][7], std_low[i][7], low[i][8], std_low[i][8], low[i][9], std_low[i][9], moderate[i][7],
                        std_moderate[i][7], moderate[i][8], std_moderate[i][8], moderate[i][9], std_moderate[i][9],
                        high[i][7], std_high[i][7], high[i][8], std_high[i][8], high[i][9], std_high[i][9],
                        averaged[i][7], std_averaged[i][7], averaged[i][8], std_averaged[i][8], averaged[i][9],
                        std_averaged[i][9], averaged[i][1], std_low[i][1]], 2).tolist()
        row.insert(0, low[i][0])
        csv.append(row)

    df = pd.DataFrame(csv)
    df.to_csv('./generated_results/' + fn + '_avg.csv', index=False, sep=',', encoding='utf-8', header=False)


def calculate_average_values(df, k, fn):
    # extract values for the three classes (split df into 3 df's)
    class_length = df.shape[0] / 3
    class_length = int(class_length)
    low, std_low = calculate_class_average(df[:class_length].reset_index(drop=True), k)
    moderate, std_moderate = calculate_class_average(df[class_length: 2 * class_length].reset_index(drop=True), k)
    high, std_high = calculate_class_average(df[2 * class_length: 3 * class_length].reset_index(drop=True), k)

    averaged, std_avg = calculate_predictor_average(low, moderate, high)
    create_csv(low, std_low, moderate, std_moderate, high, std_high, averaged, std_avg, fn)
    return df


def main():
    args = parse_arguments()
    path = args.path

    if args.k_fold <= 1:
        print('no k-fold is needed because k is <= 1')
        return 0

    if not os.path.exists('./generated_results/'):
        os.makedirs('./generated_results/')
    # process all files
    files = glob.glob(''.join([path, '/*.csv']))
    for filepath in files:
        # get filename without extension
        fn = basename(filepath)[:-4]
        df = preprocess_csv(filepath)
        df = df.fillna(0)
        print(f'Calculate average value of {fn}')
        calculate_average_values(df, args.k_fold, fn)


if __name__ == "__main__":
    main()

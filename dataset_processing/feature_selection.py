import argparse
from math import ceil
# import numpy as np
import pandas as pd
import csv
import numpy as np
import random
import scipy
from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
# from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2
# from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC
from rfpimp import permutation_importances, dropcol_importances, oob_classifier_accuracy  # feature importance
import shap  # feature importance
# from sklearn.feature_selection import VarianceThreshold  # maybe helpful to remove non relevant features
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier  # , ExtraTreesClassifier
from copy import copy
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from statistics import mean
from natsort import natsorted


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--k_fold', type=int, required=False, default=5, help='set the k for k-fold split')
    parser.add_argument('-p', '--path', type=str, required=True, help='the path of the csvs')
    parser.add_argument('-m', '--method', type=str, required=False, default='python',
                        help='choose weka for later preprocessing or preprocess with pearson')
    parser.add_argument('-f', '--filter', type=str, required=False, default='influence',
                        help='choose if threshold or amount of features is used to create subsets of features')
    parser.add_argument('-i', '--feature_importance', type=str, required=False, default='permutation',
                        help='choose the feature importance algorithm')
    args = parser.parse_args()

    return args


def preprocess_csv(csv_path, use_avg=False, kg_str='Knowledge_Gain_Level'):
    data = pd.read_csv(csv_path, encoding='utf-8')
    if use_avg:
        data = calculate_kg_level(get_vid_avg(data))

    # remove video-id, person-id and knowledge_gain because they shouldn't be selected as features
    knowledge = data[['Video_ID', kg_str]]
    data = data.drop(['Knowledge_Gain', 'Knowledge_Gain_Level'], axis=1)
    # data['Person_ID'] = data['Person_ID'].values.astype(str)
    return (data, knowledge)


def store_csv(data, name):
    nominal_to_num = {'Low': 0, 'Moderate': 1, 'High': 2}
    data.sort_values(by=['Knowledge_Gain_Level'], key=lambda x: x.map(nominal_to_num)).to_csv(
        ''.join(['./feature_selection/weka/', name, '.csv']), index=False, encoding='utf-8')


def scale_data(train_X, test_X):
    scaler = MinMaxScaler()
    scaler.fit(train_X)
    scaled_train_X, scaled_test_X = copy(scaler.transform(train_X)), copy(scaler.transform(test_X))
    return scaled_train_X, scaled_test_X


def get_vid_avg(data):
    data = copy(data.drop(['Person_ID', 'Knowledge_Gain_Level'], axis=1))
    videos = data.groupby('Video_ID')
    avg_df = None
    for video in videos.groups:
        group = videos.get_group(video)
        avg_kg = group['Knowledge_Gain'].mean()
        group_value = copy(group.head(1))
        group_value['Knowledge_Gain'] = avg_kg
        if avg_df is not None:
            avg_df = pd.concat([avg_df, group_value])
        else:
            avg_df = group_value
    return avg_df


def z_score_convert(value):
    if -0.5 <= value <= 0.5:
        return 'Moderate'
    elif value < -0.5:
        return 'Low'
    return 'High'


def calculate_kg_level(data):
    z_scores = scipy.stats.zscore(data['Knowledge_Gain'].to_numpy()).tolist()
    for i in range(len(z_scores)):
        z_scores[i] = z_score_convert(z_scores[i])
    # print(z_scores)
    data['Knowledge_Gain_Level'] = z_scores
    # print(data)
    return data


def convert_person_id(data):
    num_participants = 13
    person_id_columns = [f'Person_ID_{i + 1}' for i in range(num_participants)]
    ids = data['Person_ID'].to_numpy() - 1
    categorical = np.squeeze(np.eye(num_participants)[ids])
    categorical = pd.DataFrame(data=categorical, columns=person_id_columns).astype(int)
    data = data.drop(['Person_ID'], axis=1)
    for column in person_id_columns:
        data[column] = categorical[column].to_numpy()
    return data


def remove_redudant_features():
    return


def get_permutation_importance(name, X_train, y_train, X_test, y_test, k, store=False):
    model = RandomForestClassifier(max_depth=1, random_state=999)
    # model = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    # model = GaussianNB()
    # print(f"x-shape:{X_train.shape}, y-shape:{X_test.shape}")
    model.fit(X_train, y_train["Knowledge_Gain_Level"].to_numpy())

    r = permutation_importance(model, X_train, y_train, n_repeats=30, random_state=999,
                               scoring='accuracy')
    amount_importance = 0
    names = []
    values = []
    columns = X_train.columns.to_numpy()
    for i in abs(r.importances_mean).argsort()[::-1]:
        if r.importances_mean[i] + r.importances_std[i] != 0:
            print(f"{columns[i]} "
                  f"{r.importances_mean[i]:.3f}"
                  f" +/- {r.importances_std[i]:.3f}")
            amount_importance += 1
        names.append(columns[i])
        values.append(r.importances_mean[i])
    print(f"{amount_importance} features had an influence on the result")
    print()

    fi = pd.Series(values, names)
    if store:
        store_feature_importance(fi, name, 'permutation', k)
    return fi


def get_permutation_importance_rfpimp(name, X_train, y_train, X_test, y_test, k, store=False):
    model = RandomForestClassifier(max_depth=100, random_state=999)
    model.fit(X_train, y_train["Knowledge_Gain_Level"].to_numpy())
    fi = permutation_importances(model, X_test, y_test["Knowledge_Gain_Level"], oob_classifier_accuracy)
    fi = fi.iloc[(-fi['Importance'].abs()).argsort()]
    # fi = fi.loc[~(fi == 0).all(axis=1)]  # remove features without importance
    fi = fi.iloc[:, 0]
    if store:
        store_feature_importance(fi, name, 'permutation_rfpimp', k)
    return fi


def get_drop_column_importance(name, X_train, y_train, X_test, y_test, k, store=False):
    model = RandomForestClassifier(max_depth=100, random_state=999)
    fi = dropcol_importances(model, X_train, y_train["Knowledge_Gain_Level"].to_numpy(),
                             X_test, y_test["Knowledge_Gain_Level"].to_numpy())
    fi = fi.iloc[(-fi['Importance']).argsort()]
    fi = fi.iloc[:, 0]
    # print(fi)
    if store:
        store_feature_importance(fi, name, 'drop_column', k)
    return fi


def get_shap_importance(name, X_train, y_train, X_test, y_test, k, store=False):
    model = RandomForestClassifier(max_depth=100, random_state=999)
    model.fit(X_train, y_train["Knowledge_Gain_Level"].to_numpy())
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_train)[0].data
    columns = X_train.columns.to_list()
    # print(columns)
    # print(shap_values)
    sorted_columns = []
    sorted_importance = []
    for id in np.argsort(shap_values):
        sorted_columns.append(columns[id])
        sorted_importance.append(shap_values[id])

    fi = pd.Series(sorted_importance, sorted_columns)
    if store:
        store_feature_importance(fi, name, 'shap', k)
    return fi


def get_pearson_correlation(name, x_train, x_test, y_train, y_test, k, store=False):
    # add y-values to x-values for pearson correlation check
    x_train.insert(len(x_train.columns), 'Knowledge_Gain_Level', y_train, True)
    # get pearson correlation from x-features to y-classes
    pearson = x_train.corr()['Knowledge_Gain_Level'][:-1]
    pearson = pearson.sort_values(ascending=False)
    if store:
        store_feature_importance(pearson, name, 'pearson', k)
    return pearson


def store_feature_importance(importance, name, method, k):
    importance.name = 'Feature Importance'
    store_fi = importance.reindex(importance.sort_values(ascending=False).index)
    store_fi.to_csv(f'./feature_selection/feature_importance/{name}_{method}_correlation_set{k + 1}_features.csv',
                    index=True,
                    encoding='utf-8')


def get_feature_importance(name, X_train, X_test, y_train, y_test, k, importance_method, store=False):
    # make correct form for feature importance calculation
    if 'Person_ID' in X_train:
        X_train_corr, X_test_corr = copy(X_train).drop(['Person_ID'], axis=1), copy(X_test).drop(['Person_ID'], axis=1)
    else:
        X_train_corr, X_test_corr = copy(X_train), copy(X_test)
    # print(len(X_train_corr.columns))
    y_train_corr, y_test_corr = copy(y_train), copy(y_test)
    # convert y-classes to integers for pearson correlation
    dict_kg = {'Low': 0, 'Moderate': 1, 'High': 2}
    y_train_corr.replace(dict_kg, inplace=True)
    y_test_corr.replace(dict_kg, inplace=True)
    # choose method
    if importance_method == 'permutation':
        return get_permutation_importance(name, X_train_corr, y_train_corr, X_test_corr, y_test_corr, k, store)
    elif importance_method == 'drop_column':
        return get_drop_column_importance(name, X_train_corr, y_train_corr, X_test_corr, y_test_corr, k, store)
    elif importance_method == 'shap':
        return get_shap_importance(name, X_train_corr, y_train_corr, X_test_corr, y_test_corr, k, store)
    # default: pearson correlation
    elif importance_method == 'permutation_rfpimp':
        return get_permutation_importance_rfpimp(name, X_train_corr, y_train_corr, X_test_corr, y_test_corr, k, store)
    return get_pearson_correlation(name, X_train_corr, X_test_corr, y_train_corr, y_test_corr, k, store)


def store_avg_importance(name, mean_values, fi_method, data_method):
    if fi_method != 'pearson':
        store_p = mean_values.reindex(mean_values.sort_values(ascending=False).index)
    else:
        store_p = mean_values.reindex(mean_values.abs().sort_values(ascending=False).index)
    store_p.to_csv(f'./feature_selection/feature_importance/{name}_avg_{fi_method}_{data_method}_importance.csv',
                   index=True,
                   encoding='utf-8')


def get_non_relevant_features(data_x, data_y):
    old_columns = data_x.columns
    relevant_data_x = copy(data_x.loc[:, (data_x != data_x.iloc[0]).any()])
    removed_columns = [column for column in old_columns if column not in relevant_data_x.columns]
    print(f'Amount of non relevant features is {len(removed_columns)}')
    print(removed_columns)
    return removed_columns


def create_filtered_sets_best_features(name, fi, x_train, x_test, y_train, y_test, k):
    max_features = None
    all_features = [374, 350, 300, 250, 200, 150, 100, 90, 80, 70, 60, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 1]
    text_features = [337, 300, 250, 200, 150, 100, 90, 80, 70, 60, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 1]
    multimedia_features = [37, 35, 30, 25, 20, 15, 10, 5, 1]

    # remove one hot vector of person_id from feature importance (person_id should stay as one complete vector)
    specific_idx = [index for index in fi.index if 'Person_ID' in index]
    if specific_idx:
        fi = fi.drop(specific_idx)
    specific_idx = natsorted(specific_idx)
    # get correct list of feature-selection
    if name == 'text_features':
        max_features = text_features
    elif name == 'multimedia_features':
        max_features = multimedia_features
    else:
        max_features = all_features
    # sort by absolute value
    fi = fi.reindex(fi.abs().sort_values(ascending=False).index)
    # create csv-files with maximum amount of features
    for max_feature in max_features:
        features = list(fi[0:max_feature].index)
        # store csv-files
        if specific_idx:
            features = features + ['Person_ID']
        filtered_train = copy(x_train[features])
        filtered_test = copy(x_test[features])

        max_string = ''.join([str(max_feature), 'features'])
        store_filtered(name, filtered_train, y_train, filtered_test, y_test, max_string, k)


def create_filtered_sets_threshold(name, fi, x_train, x_test, y_train, y_test, k):
    # get highest correlation to stop loop
    fi = fi.sort_values(ascending=False)
    stop_i = int(ceil(fi[0] * 100)) if fi[0] >= abs(fi[-1]) else int(ceil(abs(fi[-1]) * 100))
    # start creating subsets
    previous = -1
    for threshold in range(0, stop_i):
        threshold = threshold * 0.01
        features = list(fi[abs(fi) > threshold].index)
        # stop if there are no features
        if not len(features):
            break
        # don't make duplicate subsets
        if len(features) == previous:
            continue
        previous = len(features)
        # store csv-files
        filtered_train = copy(x_train[features])
        filtered_test = copy(x_test[features])

        if threshold == 0.0:
            threshold = 0
        th_str = ''.join([str(threshold), 'threshold'])
        store_filtered(name, filtered_train, y_train, filtered_test, y_test, th_str, k)


def create_filtered_sets_influence(name, fis, x_train, x_test, y_train, y_test, k, fi_method, data_method):
    features = [column for column in fis[fis >= 0.0].index.to_list() if 'Person_ID' not in column]
    if [column for column in x_train.columns if 'Person_ID' in column]:
        features = features + ['Person_ID']
    print(f'Amount of features: {len(features)}')
    filtered_train = copy(x_train[features])
    filtered_test = copy(x_test[features])
    store_filtered(name, filtered_train, y_train, filtered_test, y_test, 'influence', k, fi_method, data_method)


def create_embedding_csvs(slide_embd_train, y_train, slide_embd_test, y_test, transcript_embd_train,
                          transcript_embd_test, k, data_method):
    nominal_to_num = {'Low': 0, 'Moderate': 1, 'High': 2}
    # only slide embedding
    pd.concat([slide_embd_train, y_train], axis=1).sort_values(by=['Knowledge_Gain_Level'],
                                                               key=lambda x: x.map(nominal_to_num)).to_csv(''.join(
        ['./feature_selection/pearson/train/embedding_', str(k + 1), '_', data_method, '_only_slide.csv']),
        index=False, encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)
    pd.concat([slide_embd_test, y_test], axis=1).sort_values(by=['Knowledge_Gain_Level'],
                                                             key=lambda x: x.map(nominal_to_num)).to_csv(
        ''.join(
            ['./feature_selection/pearson/test/embedding_', str(k + 1), '_', data_method, '_only_slide_test.csv']),
        index=False, encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)
    # only transcript embedding
    pd.concat([transcript_embd_train, y_train], axis=1).sort_values(by=['Knowledge_Gain_Level'],
                                                                    key=lambda x: x.map(nominal_to_num)).to_csv(''.join(
        ['./feature_selection/pearson/train/embedding_', str(k + 1), '_', data_method, '_only_transcript.csv']),
        index=False, encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)
    pd.concat([transcript_embd_test, y_test], axis=1).sort_values(by=['Knowledge_Gain_Level'],
                                                                  key=lambda x: x.map(nominal_to_num)).to_csv(
        ''.join(
            ['./feature_selection/pearson/test/embedding_', str(k + 1), '_', data_method, '_only_transcript_test.csv']),
        index=False, encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)
    # only both embeddings
    pd.concat(
        [slide_embd_train, transcript_embd_train.drop(list(transcript_embd_train.filter(regex='Person_ID')), axis=1),
         y_train], axis=1).sort_values(by=['Knowledge_Gain_Level'], key=lambda x: x.map(nominal_to_num)).to_csv(''.join(
        ['./feature_selection/pearson/train/embedding_', str(k + 1), '_', data_method, '_both_embd.csv']),
        index=False, encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)
    pd.concat([slide_embd_test, transcript_embd_test.drop(list(transcript_embd_test.filter(regex='Person_ID')), axis=1),
               y_test], axis=1).sort_values(by=['Knowledge_Gain_Level'], key=lambda x: x.map(nominal_to_num)).to_csv(
        ''.join(
            ['./feature_selection/pearson/test/embedding_', str(k + 1), '_', data_method, '_both_embd_test.csv']),
        index=False, encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)


def store_filtered(name, filtered_train, y_train, filtered_test, y_test, specific, k, fi_method, data_method):
    if 'Person_ID' in filtered_train:
        filtered_train = convert_person_id(filtered_train)
    if 'Person_ID' in filtered_test:
        filtered_test = convert_person_id(filtered_test)
    nominal_to_num = {'Low': 0, 'Moderate': 1, 'High': 2}
    pd.concat([filtered_train, y_train], axis=1).sort_values(by=['Knowledge_Gain_Level'],
                                                             key=lambda x: x.map(nominal_to_num)).to_csv(''.join(
        ['./feature_selection/pearson/train/', name, '_', str(k + 1), '_', fi_method, '_', data_method, '_', specific,
         '_without.csv']),
        index=False, encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)
    pd.concat([filtered_test, y_test], axis=1).sort_values(by=['Knowledge_Gain_Level'],
                                                           key=lambda x: x.map(nominal_to_num)).to_csv(
        ''.join(
            ['./feature_selection/pearson/test/', name, '_', str(k + 1), '_', fi_method, '_', data_method, '_',
             specific,
             '_without_test.csv']),
        index=False, encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)


def embedding_split(k, slide_embd, transcript_embd):
    fold_dict = {}
    num_participants = 13
    person_cols = [f'Person_ID_{i + 1}' for i in range(num_participants)]
    video_ids = np.array(['1_2a', '1_2b', '1_2c', '1_2d', '1_3a', '1_3b', '1_3c', '2_2a', '2_2b', '2_2c',
                          '2_2d', '3_2b', '3_3a', '3_3b', '4_2a', '4_3a', '5_1a', '5_1b', '6_2a', '7_2b', '7_3a',
                          '7_3c'])
    k_fold = KFold(n_splits=k, shuffle=True, random_state=1234)
    for i, (train, test) in enumerate(k_fold.split(video_ids)):
        train = video_ids[train]
        test = video_ids[test]
        slide_train, slide_test = copy(slide_embd[0].iloc[np.where(slide_embd[0]['Video_ID'].isin(train))]), \
                                  copy(slide_embd[0].iloc[np.where(slide_embd[0]['Video_ID'].isin(test))])
        slide_train, slide_test = slide_train.drop(['Video_ID'], axis=1), slide_test.drop(['Video_ID'], axis=1)
        transcript_train, transcript_test = copy(
            transcript_embd[0].iloc[np.where(transcript_embd[0]['Video_ID'].isin(train))]), \
                                            copy(transcript_embd[0].iloc[
                                                     np.where(transcript_embd[0]['Video_ID'].isin(test))])
        transcript_train, transcript_test = transcript_train.drop(['Video_ID'], axis=1), transcript_test.drop(
            ['Video_ID'], axis=1)
        slide_y_train, slide_y_test = copy(slide_embd[1].iloc[np.where(slide_embd[1]['Video_ID'].isin(train))]), \
                                      copy(slide_embd[1].iloc[np.where(slide_embd[1]['Video_ID'].isin(test))])
        slide_y_train, slide_y_test = slide_y_train.drop(['Video_ID'], axis=1), slide_y_test.drop(['Video_ID'], axis=1)
        transcript_y_train, transcript_y_test = copy(
            transcript_embd[1].iloc[np.where(transcript_embd[1]['Video_ID'].isin(train))]), \
                                                copy(transcript_embd[1].iloc[
                                                         np.where(transcript_embd[1]['Video_ID'].isin(test))])
        transcript_y_train, transcript_y_test = transcript_y_train.drop(['Video_ID'], axis=1), \
                                                transcript_y_test.drop(['Video_ID'], axis=1)
        slide_train = convert_person_id(slide_train)
        slide_test = convert_person_id(slide_test)
        transcript_train = convert_person_id(transcript_train)
        transcript_test = convert_person_id(transcript_test)
        # scale data
        slide_train.loc[:, slide_train.columns != 'Person_ID'], slide_test.loc[:, slide_test.columns != 'Person_ID'] \
            = scale_data(slide_train.loc[:, slide_train.columns != 'Person_ID'],
                         slide_test.loc[:, slide_test.columns != 'Person_ID'])
        transcript_train.loc[:, transcript_train.columns != 'Person_ID'], transcript_test.loc[:, transcript_train.columns != 'Person_ID'] \
            = scale_data(transcript_train.loc[:, transcript_train.columns != 'Person_ID'],
                         transcript_test.loc[:, transcript_test.columns != 'Person_ID'])
        # store values
        create_embedding_csvs(slide_train, slide_y_train, slide_test, slide_y_test, transcript_train,
                              transcript_test, i, 'persons')


def avg_split(k, name, feature, filter_type, slide_embd, transcript_embd, non_relevant, importance_method):
    # randomly split data into train and test
    print(f'Split average of {name}')
    #random.seed(1234)
    fis = []  # feature importances
    sets = []
    k_fold = StratifiedKFold(n_splits=k, shuffle=True, random_state=1234)
    for i, (train, test) in enumerate(k_fold.split(feature[0], feature[1]['Knowledge_Gain_Level'])):
        print(f'Calculate split{i + 1}')
        # sort videos into train and test
        x_train, x_test = copy(feature[0].iloc[train]), copy(feature[0].iloc[test])
        y_train, y_test = copy(feature[1].iloc[train]), copy(feature[1].iloc[test])
        #print(train, test)
        # remove non relevant features
        non_relevant = ['Video_ID'] + non_relevant
        x_train = x_train.drop([column for column in non_relevant if column in x_train.columns], axis=1)
        x_test = x_test.drop([column for column in non_relevant if column in x_test.columns], axis=1)
        y_train, y_test = y_train.drop(['Video_ID'], axis=1), y_test.drop(['Video_ID'], axis=1)
        # split embeddings into train and test
        slide_x_train, slide_x_test = copy(slide_embd[0].iloc[train]), copy(slide_embd[0].iloc[test])
        slide_x_train, slide_x_test = slide_x_train.drop(['Video_ID'], axis=1), slide_x_test.drop(['Video_ID'], axis=1)
        transcript_x_train, transcript_x_test = copy(transcript_embd[0].iloc[train]), copy(
            transcript_embd[0].iloc[test])
        transcript_x_train, transcript_x_test = transcript_x_train.drop(['Video_ID'], axis=1), transcript_x_test.drop(
            ['Video_ID'], axis=1)
        slide_y_train, slide_y_test = copy(slide_embd[1].iloc[train]), copy(slide_embd[1].iloc[test])
        slide_y_train, slide_y_test = slide_y_train.drop(['Video_ID'], axis=1), slide_y_test.drop(['Video_ID'], axis=1)
        transcript_y_train, transcript_y_test = copy(transcript_embd[1].iloc[train]), copy(
            transcript_embd[1].iloc[test])
        transcript_y_train, transcript_y_test = transcript_y_train.drop(['Video_ID'], axis=1), \
                                                transcript_y_test.drop(['Video_ID'], axis=1)
        # scale data
        x_train[:], x_test[:] = scale_data(x_train, x_test)
        slide_x_train[:], slide_x_test[:] = scale_data(slide_x_train, slide_x_test)
        transcript_x_train[:], transcript_x_test[:] = scale_data(transcript_x_train, transcript_x_test)

        sets.append((x_train, x_test, y_train, y_test))
        fi = get_feature_importance(name, x_train, x_test, y_train, y_test, i, importance_method, True)
        fis.append(fi)
        if filter_type == "threshold":
            create_filtered_sets_threshold(name, fi, x_train, x_test, y_train, y_test, i)
        elif filter_type == "amount":
            create_filtered_sets_best_features(name, fi, x_train, x_test, y_train, y_test, i)
        elif filter_type == "influence":
            create_filtered_sets_influence(name, fi, x_train, x_test, y_train, y_test, i, importance_method,
                                           'videos')
        # store embedding features
        create_embedding_csvs(slide_x_train, slide_y_train, slide_x_test, slide_y_test, transcript_x_train,
                              transcript_x_test, i, 'videos')
    concated = pd.concat(fis)
    mean_values = concated.groupby(concated.index).mean()
    store_avg_importance(name, mean_values, importance_method, 'videos')


def data_split(k, name, feature, filter_type, slide_embd, transcript_embd, non_relevant, importance_method):
    num_participants = 13
    #person_cols = [f'Person_ID_{i + 1}' for i in range(num_participants)]
    k_fold = KFold(n_splits=5, random_state=1234, shuffle=True)
    video_ids = np.array(['1_2a', '1_2b', '1_2c', '1_2d', '1_3a', '1_3b', '1_3c', '2_2a', '2_2b', '2_2c',
                          '2_2d', '3_2b', '3_3a', '3_3b', '4_2a', '4_3a', '5_1a', '5_1b', '6_2a', '7_2b', '7_3a',
                          '7_3c'])
    # iterate through features to calculate correlation/feature importance
    print(f'\nGenerate splits for csv-file: {name}')
    complete_fi = []
    sets = []
    for i, (train, test) in enumerate(k_fold.split(video_ids)):
        print(f'Calculate split{i + 1}')
        # get train/test for features x-values and y-values
        train = video_ids[train]
        test = video_ids[test]
        # sort videos into train and test
        x_train, x_test = copy(feature[0].iloc[np.where(feature[0]['Video_ID'].isin(train))]), copy(
            feature[0].iloc[np.where(feature[0]['Video_ID'].isin(test))])
        # remove non relevant features
        non_relevant = ['Video_ID'] + non_relevant
        x_train = x_train.drop([column for column in non_relevant if column in x_train.columns], axis=1)
        x_test = x_test.drop([column for column in non_relevant if column in x_test.columns], axis=1)
        y_train, y_test = copy(feature[1].iloc[np.where(feature[1]['Video_ID'].isin(train))]), copy(
            feature[1].iloc[np.where(feature[1]['Video_ID'].isin(test))])
        y_train, y_test = y_train.drop(['Video_ID'], axis=1), y_test.drop(['Video_ID'], axis=1)
        # scale data
        x_train.loc[:, x_train.columns != 'Person_ID'], x_test.loc[:, x_test.columns != 'Person_ID'] \
            = scale_data(x_train.loc[:, x_train.columns != 'Person_ID'],
                         x_test.loc[:, x_test.columns != 'Person_ID'])
        #print(x_train)
        # get feature importance
        fis = get_feature_importance(name, convert_person_id(x_train), convert_person_id(x_test), y_train, y_test, i,
                                     importance_method, True).fillna(0)
        sets.append((x_train, x_test, y_train, y_test))
        complete_fi.append(copy(fis))
        if filter_type == "threshold":
            create_filtered_sets_threshold(name, fis, x_train, x_test, y_train, y_test, i)
        elif filter_type == "amount":
            create_filtered_sets_best_features(name, fis, x_train, x_test, y_train, y_test, i)
        elif filter_type == "influence":
            create_filtered_sets_influence(name, fis, x_train, x_test, y_train, y_test, i, importance_method,
                                           'persons')
    concated = pd.concat(complete_fi)
    mean_values = concated.groupby(concated.index).mean()
    store_avg_importance(name, mean_values, importance_method, 'persons')


def create_csvs_weka(k, name, feature):
    video_ids = ['1_2a', '1_2b', '1_2c', '1_2d', '1_3a', '1_3b', '1_3c', '2_2a', '2_2b', '2_2c',
                 '2_2d', '3_2b', '3_3a', '3_3b', '4_2a', '4_3a', '5_1a', '5_1b', '6_2a', '7_2b', '7_3a', '7_3c']
    k_fold = KFold(n_splits=k, shuffle=True, random_state=1234)
    # print(k_fold.split(video_ids))
    for i, (train, test) in enumerate(k_fold.split(feature[0], feature[1])):
        x_train, x_test = copy(feature[0].iloc[train]), copy(feature[0].iloc[test])
        y_train, y_test = copy(feature[1][train]), copy(feature[1][test])
        x_train.insert(len(x_train.columns), 'Knowledge_Gain_Level', y_train, True)
        x_test.insert(len(x_test.columns), 'Knowledge_Gain_Level', y_test, True)
        # store train and test set
        store_csv(x_train, ''.join(['train/', name, '_', str(i + 1), '_weka_features']))
        store_csv(x_test, ''.join(['test/', name, '_', str(i + 1), '_weka_features', '_test']))


def show_kg_prediction_stats(class_vals, overall, y_true, y_pred):
    dict_kg = {0: 'Low', 1: 'Moderate', 2: 'High'}
    sorted_class_vals = [[], [], []]
    for class_val in class_vals[:-1]:
        for i, val in enumerate(class_val):
            sorted_class_vals[i].append(val)
    for i, class_val in enumerate(sorted_class_vals):
        print(f'{dict_kg[i]}:')
        print(f'Precision:{round(class_val[0], 2)}, Recall:{round(class_val[1], 2)}, F1:{round(class_val[2], 2)}\n')
    print('Overall:')
    print(f'Precision:{round(overall[0], 2)}, Recall:{round(overall[1], 2)}, F1:{round(overall[2], 2)}')
    print(f'Accuracy:{round(accuracy_score(y_true, y_pred) * 100, 2)}')


def user_kg_prediction(name, feature):
    kg_class = feature[1]
    # kg_class = kg_class.drop(['Video_ID'], axis=1)
    z_scores = scipy.stats.zscore(kg_class['Knowledge_Gain'].to_numpy()).tolist()
    kg_class['Knowledge_Gain'] = z_scores
    # create groups of person_ids
    person_ids = copy(pd.concat([feature[0]['Person_ID'], kg_class], axis=1))
    grouped_ids = person_ids.groupby(['Person_ID'], as_index=False)
    y_pred = []
    y_true = []
    for name, group in grouped_ids:
        amount = len(group)
        group = group.reset_index(drop=True)
        for i in range(amount):
            test_value = group.iloc[i]['Knowledge_Gain']
            predicted = group.drop([i])['Knowledge_Gain'].mean()
            # predicted = group['Knowledge_Gain'].mean()
            # convert to three classes
            test_value = z_score_convert(test_value)
            predicted = z_score_convert(predicted)
            y_true.append(test_value)
            y_pred.append(predicted)
    class_vals = list(precision_recall_fscore_support(y_true, y_pred, average=None, labels=['Low', 'Moderate', 'High']))
    overall = list(precision_recall_fscore_support(y_true, y_pred, average='macro'))
    overall[2] = 2 / (1 / overall[0] + 1 / overall[1])
    show_kg_prediction_stats(class_vals, overall, y_true, y_pred)


def video_kg_prediction(name, feature, avg_feature, mode='avg'):
    kg_class = feature[1]
    kg_class_avg = avg_feature[1]
    if mode == 'avg':
        z_scores = scipy.stats.zscore(kg_class['Knowledge_Gain'].to_numpy()).tolist()
        kg_class['Knowledge_Gain'] = z_scores
        z_scores_avg = scipy.stats.zscore(kg_class_avg['Knowledge_Gain'].to_numpy()).tolist()
        kg_class_avg['Knowledge_Gain'] = z_scores_avg
    else:
        random.seed(1234)
    # group person ids to match with video_id value
    person_ids = copy(pd.concat([feature[0]['Person_ID'], kg_class], axis=1))
    grouped_persons = person_ids.groupby(['Person_ID'], as_index=False)
    # scoring values
    y_true = []
    y_pred = []
    for id, row in kg_class_avg.iterrows():
        video_id = row['Video_ID']
        video_kg_class = z_score_convert(row['Knowledge_Gain']) if mode == 'avg' else row['Knowledge_Gain_Level']
        y_preds = []
        for _, group in grouped_persons:
            other_vid_value = None
            if video_id in group['Video_ID'].values:
                if mode == 'avg':
                    other_vid_value = group[group['Video_ID'] != video_id]['Knowledge_Gain'].mean()
                else:
                    other_vid_values = group[group['Video_ID'] != video_id]['Knowledge_Gain_Level'].mode()
                    amount_most = len(other_vid_values)
                    if amount_most >= 2:
                        most_id = random.randint(0, amount_most - 1)
                        other_vid_value = other_vid_values[most_id]
                    else:
                        other_vid_value = other_vid_values[0]
                y_preds.append(other_vid_value)
        if mode == 'avg':
            final_pred = z_score_convert(np.mean(y_preds))
        else:
            unique_counts = np.unique(y_preds, return_counts=True)
            max_val = unique_counts[1][np.argmax(unique_counts[1])]
            max_entries = []
            for i, value in enumerate(unique_counts[1]):
                if value == max_val:
                    max_entries.append(unique_counts[0][i])
            amount_most = len(max_entries)
            if amount_most >= 2:
                most_id = random.randint(0, amount_most - 1)
                final_pred = max_entries[most_id]
            else:
                final_pred = max_entries[0]
        y_true.append(video_kg_class)
        y_pred.append(final_pred)
    class_vals = list(precision_recall_fscore_support(y_true, y_pred, average=None, labels=['Low', 'Moderate', 'High']))
    overall = list(precision_recall_fscore_support(y_true, y_pred, average='macro'))
    overall[2] = 2 / (1 / overall[0] + 1 / overall[1])
    show_kg_prediction_stats(class_vals, overall, y_true, y_pred)


def main():
    args = parse_arguments()
    path = args.path

    # check if commands are correct
    if args.method.lower() not in ['python', 'weka', 'avg_python', 'person_id', 'video_id1', 'video_id2', 'video_id3']:
        print('The choosen method has to be python, avg_python, person_id or weka')
        return
    if args.filter.lower() not in ['amount', 'threshold', 'influence']:
        print('The choosen filter has to be amount, threshold or influence')
        return
    if args.feature_importance.lower() not in ['pearson', 'permutation', 'permutation_rfpimp', 'shap', 'drop_column']:
        print('The choosen feature importance algorithm has to be pearson, permutation, permutation_rfpimp, '
              'shap or drop_column')
        return

    fi_method = args.feature_importance
    print("run")
    if args.method.lower() == 'weka':
        print('Use weka method to get sets')
        all_features = preprocess_csv(''.join([path, 'all_features.csv']))
        text_features = preprocess_csv(''.join([path, 'text_features.csv']))
        multimedia_features = preprocess_csv(''.join([path, 'multimedia_features.csv']))
        create_csvs_weka(args.k_fold, 'all_features', all_features)
        create_csvs_weka(args.k_fold, 'text_features', text_features)
        create_csvs_weka(args.k_fold, 'multimedia_features', multimedia_features)

    elif args.method.lower() == 'python':
        print(f'Use python and the feature importance method "{args.feature_importance.lower()}" to get sets')
        all_features = preprocess_csv(''.join([path, 'all_features.csv']))
        text_features = preprocess_csv(''.join([path, 'text_features.csv']))
        multimedia_features = preprocess_csv(''.join([path, 'multimedia_features.csv']))
        slide_embd = preprocess_csv(''.join([path, 'slide_embedding.csv']))
        transcript_embd = preprocess_csv(''.join([path, 'transcript_embedding.csv']))
        non_relevant = get_non_relevant_features(all_features[0], all_features[1])
        data_split(args.k_fold, 'all_features', all_features, args.filter.lower(), slide_embd, transcript_embd,
                   non_relevant, fi_method)
        data_split(args.k_fold, 'text_features', text_features, args.filter.lower(), slide_embd, transcript_embd,
                   non_relevant, fi_method)
        data_split(args.k_fold, 'multimedia_features', multimedia_features, args.filter.lower(), slide_embd,
                   transcript_embd, non_relevant, fi_method)
        embedding_split(args.k_fold, slide_embd, transcript_embd)
    elif args.method.lower() == 'avg_python':
        print(
            f'Use avg dataset with python and the feature importance method "{args.feature_importance.lower()}" to get sets')
        all_features = preprocess_csv(''.join([path, 'all_features.csv']), True)

        text_features = preprocess_csv(''.join([path, 'text_features.csv']), True)
        multimedia_features = preprocess_csv(''.join([path, 'multimedia_features.csv']), True)
        slide_embd = preprocess_csv(''.join([path, 'slide_embedding.csv']), True)
        transcript_embd = preprocess_csv(''.join([path, 'transcript_embedding.csv']), True)
        non_relevant = get_non_relevant_features(all_features[0], all_features[1])
        avg_split(args.k_fold, 'all_features', all_features, args.filter.lower(), slide_embd, transcript_embd,
                  non_relevant,
                  fi_method)
        avg_split(args.k_fold, 'text_features', text_features, args.filter.lower(), slide_embd, transcript_embd,
                  non_relevant,
                  fi_method)
        avg_split(args.k_fold, 'multimedia_features', multimedia_features, args.filter.lower(), slide_embd,
                  transcript_embd,
                  non_relevant, fi_method)
    elif args.method.lower() == 'person_id':
        print('Calculate knowledge gain prediction with users only')
        all_features = preprocess_csv(''.join([path, 'all_features.csv']), kg_str='Knowledge_Gain')
        user_kg_prediction('person_id', all_features)
    elif args.method.lower() == 'video_id1':
        all_features = preprocess_csv(''.join([path, 'all_features.csv']), kg_str='Knowledge_Gain')
        all_features_avg = preprocess_csv(''.join([path, 'all_features.csv']), True, kg_str='Knowledge_Gain')
        video_kg_prediction('video_id1', all_features, all_features_avg)
    elif args.method.lower() == 'video_id2':
        all_features = preprocess_csv(''.join([path, 'all_features.csv']), kg_str='Knowledge_Gain_Level')
        all_features_avg = preprocess_csv(''.join([path, 'all_features.csv']), True, kg_str='Knowledge_Gain_Level')
        video_kg_prediction('video_id2', all_features, all_features_avg, 'majority')


if __name__ == "__main__":
    main()

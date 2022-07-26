import sys
import csv
import pandas as pd


def merge_csv(first, second, result):
    """ Merges two files together and delete redundant columns.
    :param first: the first file
    :param second: the second file
    :param result: the combined result
    :return: merged csv-file
    """
    first_csv = csv.reader(first, delimiter=',')
    second_csv = csv.reader(second, delimiter=',')
    first_arr = []
    second_arr = []
    for row in first_csv:
        first_arr.append(row)
    for row in second_csv:
        second_arr.append(row)
    if len(first_arr) != len(second_arr):  # files must have the same row length (because they should be from same data)
        print("No same row length")
        first.close()
        second.close()
        sys.exit(1)
    else:
        writer = csv.writer(result, delimiter=',')
        i = 0
        while i < len(first_arr): # deletes de knowledge_gain_value from the first file and appends the second file
            modified_row = first_arr[i][:-1] + second_arr[i]
            writer.writerow(modified_row)
            i += 1
        result.close()
        first.close()
        second.close()


def merge_df(first, second, name):
    first_df = pd.read_csv(first, encoding='utf-8').drop(['Knowledge_Gain_Level'], axis=1)
    second_df = pd.read_csv(second, encoding='utf-8')
    second_df = pd.read_csv(second, encoding='utf-8').drop(list(second_df.filter(regex ='Person_ID')), axis=1)
    merged = pd.concat([first_df, second_df], axis=1)
    if 'test' in name:
        merged.to_csv(f'./merged/merged_embeddings/test/{name}',
            index=False, encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)
    else:
        merged.to_csv(f'./merged/merged_embeddings/train/{name}',
            index=False, encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)


if __name__ == "__main__":
    for path, end in [('./feature_selection/pearson/train/', ''), ('./feature_selection/pearson/test/', '_test')]:
        for i in range(5):
            i = i+1
            '''
            merge_df(f'{path}all_features_{i}_drop_column_persons_influence_without{end}.csv', f'{path}embedding_{i}_persons_both_embd{end}.csv', f'all_features_{i}_persons_both_embed{end}.csv')
            merge_df(f'{path}multimedia_features_{i}_drop_column_persons_influence_without{end}.csv', f'{path}embedding_{i}_persons_both_embd{end}.csv', f'multimedia_features_{i}_persons_both_embed{end}.csv')
            merge_df(f'{path}text_features_{i}_drop_column_persons_influence_without{end}.csv', f'{path}embedding_{i}_persons_both_embd{end}.csv', f'text_features_{i}_persons_both_embed{end}.csv')

            merge_df(f'{path}all_features_{i}_drop_column_persons_influence_without{end}.csv', f'{path}embedding_{i}_persons_only_slide{end}.csv', f'all_features_{i}_persons_slides_embed{end}.csv')
            merge_df(f'{path}multimedia_features_{i}_drop_column_persons_influence_without{end}.csv', f'{path}embedding_{i}_persons_only_slide{end}.csv', f'multimedia_features_{i}_persons_slides_embed{end}.csv')
            merge_df(f'{path}text_features_{i}_drop_column_persons_influence_without{end}.csv', f'{path}embedding_{i}_persons_only_slide{end}.csv', f'text_features_{i}_persons_slides_embed{end}.csv')
            
            merge_df(f'{path}all_features_{i}_drop_column_persons_influence_without{end}.csv', f'{path}embedding_{i}_persons_only_transcript{end}.csv', f'all_features_{i}_persons_srt_embed{end}.csv')
            merge_df(f'{path}multimedia_features_{i}_drop_column_persons_influence_without{end}.csv', f'{path}embedding_{i}_persons_only_transcript{end}.csv', f'multimedia_features_{i}_persons_srt_embed{end}.csv')
            merge_df(f'{path}text_features_{i}_drop_column_persons_influence_without{end}.csv', f'{path}embedding_{i}_persons_only_transcript{end}.csv', f'text_features_{i}_persons_srt_embed{end}.csv')
            # average results
            '''
            merge_df(f'{path}all_features_{i}_drop_column_videos_influence_without{end}.csv', f'{path}embedding_{i}_videos_both_embd{end}.csv', f'all_features_{i}_videos_both_embed{end}.csv')
            merge_df(f'{path}multimedia_features_{i}_drop_column_videos_influence_without{end}.csv', f'{path}embedding_{i}_videos_both_embd{end}.csv', f'multimedia_features_{i}_videos_both_embed{end}.csv')
            merge_df(f'{path}text_features_{i}_drop_column_videos_influence_without{end}.csv', f'{path}embedding_{i}_videos_both_embd{end}.csv', f'text_features_{i}_videos_both_embed{end}.csv')

            merge_df(f'{path}all_features_{i}_drop_column_videos_influence_without{end}.csv', f'{path}embedding_{i}_videos_only_slide{end}.csv', f'all_features_{i}_videos_slides_embed{end}.csv')
            merge_df(f'{path}multimedia_features_{i}_drop_column_videos_influence_without{end}.csv', f'{path}embedding_{i}_videos_only_slide{end}.csv', f'multimedia_features_{i}_videos_slides_embed{end}.csv')
            merge_df(f'{path}text_features_{i}_drop_column_videos_influence_without{end}.csv', f'{path}embedding_{i}_videos_only_slide{end}.csv', f'text_features_{i}_videos_slides_embed{end}.csv')
            
            merge_df(f'{path}all_features_{i}_drop_column_videos_influence_without{end}.csv', f'{path}embedding_{i}_videos_only_transcript{end}.csv', f'all_features_{i}_videos_srt_embed{end}.csv')
            merge_df(f'{path}multimedia_features_{i}_drop_column_videos_influence_without{end}.csv', f'{path}embedding_{i}_videos_only_transcript{end}.csv', f'multimedia_features_{i}_videos_srt_embed{end}.csv')
            merge_df(f'{path}text_features_{i}_drop_column_videos_influence_without{end}.csv', f'{path}embedding_{i}_videos_only_transcript{end}.csv', f'text_features_{i}_videos_srt_embed{end}.csv')

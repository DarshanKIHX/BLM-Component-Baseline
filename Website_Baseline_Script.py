
import os
import pandas as pd
import json
import re
from tqdm import tqdm
import numpy as np
import os
import traceback
from get_metadata import clean_value_text 
from get_metadata import process_metadata_final
from get_metadata import get_metadata
import ast


gt_df=pd.read_csv('enter the path to ground truth csv file')

def extract_comment(x):
    try:
        d = ast.literal_eval(x)  # Safely parse string to dict
        return d.get('comment') if isinstance(d, dict) else None
    except (ValueError, SyntaxError):
        return None  # Return None if parsing fails

gt_df['comment_value'] = gt_df['metadata'].apply(extract_comment)


rel_gt=gt_df[['claim_id','document_id','key_text','value_text','page_id','comment_value']]

#
blm_output_dir = '/mnt/ihxaidata01/nithilanv/parakeet_annotated_data/blm_outputs'


blm_output_files = os.listdir(blm_output_dir)

blm_output_paths = [os.path.join(blm_output_dir, blm_filename) 
                    for blm_filename in blm_output_files]

pr2 = pd.DataFrame()

mapper=pd.read_csv('/home/darshank/workspace/mapper/IHX_BLM_recno_updated_mapper.csv')

def get_header_objects(df):
    """
    Gets all individual header objects from a dataframe.
    Args:
        df: BLM raw dataframe
    Returns:
        aggregated_df: Constructed BLM dataframe with predicted base labels as the columns
    """
    df['base_label'] = df['label'].str.replace(r"^[BIES]-", "", regex=True)
    aggregated_data = []
    grouped = df.groupby('line_id')
    for line_id, group in grouped:

        line_data = {'line_id': line_id}

        for label in group['base_label'].unique():
            label_text = " ".join(group[group['base_label'] == label]['text'].tolist())
            line_data[label] = label_text
        aggregated_data.append(line_data)
    aggregated_df = pd.DataFrame(aggregated_data)

    header_objects = []
        
    if 'Header' in aggregated_df.columns:
            header_objects = aggregated_df['Header'].dropna().tolist()

    return header_objects

we_df = []
error_dict = {}
for path in tqdm(blm_output_paths):
    try:
        txn_id = (path.split('_')[-3].split('/')[-1])
        page_id = (path.split('_')[-1].split('.')[0])
        if not txn_id in rel_gt['document_id'].tolist():
            print('meow')
            continue
        web = None

        with open(path, 'r') as file:
            data = json.load(file)
            
        meta = data.get('meta_data')
        if not meta:
            continue
        df = pd.DataFrame(meta)

        raw = pd.DataFrame(data.get('ocr+lm_raw_output', []))
        header = raw[raw['label'].str.contains('Header')].copy()

        header_objects = get_header_objects(header)

        final_meta = process_metadata_final(get_metadata(df), mapper, pr2)

        web = final_meta.loc[final_meta['standard_label'] == 'Website', 'processed_ner'].tolist()
        picked_up_keys = final_meta.loc[final_meta['standard_label'] == 'Website', 'ner_key'].tolist()

    except Exception as e:
        traceback.print_exc()
        print(f"Error processing {path}: {e}")
        error_dict[path] = e
        continue

    temp_dict = {
        'txn_id': txn_id,
        'page_id': page_id,
        'website': web,
        'picked_key': picked_up_keys,
        'header_objects': header_objects,
        'header_rows': header 
    }

    we_df.append(temp_dict)

gn_df= pd.DataFrame(we_df)

import re

def extract_websites(text_lines):
    """
    Extracts website URLs from a list of text strings.

    Args:
        text_lines (list of str): List of text lines.

    Returns:
        list of str: List of extracted website URLs.
    """
    url_pattern = re.compile(r'(https?:\/\/[^\s]+|www\.[^\s]+)')
    combined_text = " ".join(text_lines)
    matches = url_pattern.findall(combined_text)
    return matches

gn_df['extr_webs'] = gn_df['header_objects'].apply(extract_websites)

gn_df['clext'] = gn_df['extr_webs'].apply(lambda x: x[0] if x else None)

extr_df = gn_df[['txn_id', 'page_id', 'extr_webs', 'clext']]

def extract_comment(x):
    try:
        d = ast.literal_eval(x)  # Safely parse string to dict
        return d.get('comment') if isinstance(d, dict) else None
    except (ValueError, SyntaxError):
        return None  # Return None if parsing fails

gt_df['comment_value'] = gt_df['metadata'].apply(extract_comment)

gn_gt= gt_df[['document_id', 'page_id', 'value_text', 'claim_id','comment_value','status']]

gn_gt.rename(columns={'document_id': 'txn_id'}, inplace=True)
gn_gt.rename(columns={'value_text': 'GT_Web'}, inplace=True)

extr_df['key'] = gn_df.apply(lambda row: row['txn_id'] + '_' + str(row['page_id']), axis = 1)

gn_gt['key'] = gn_gt.apply(lambda row: row['txn_id'] + '_' + str(row['page_id']), axis = 1)

combined_df = gn_gt.merge(extr_df, how='left', on='key')

combined_df['match'] = combined_df.apply(lambda row: row['GT_Web'] == row['clext'], axis = 1)

def evaluate_date_match(gt, pred):
    """
    Evaluate match between ground truth and predicted gender.
    
    Returns:
        'True'           -> both not None/NaN and match (male/m/f == male/m/f)
        'False'          -> both not None/NaN but do not match
        'False Positive' -> GT missing, prediction exists
        'False Negative' -> GT exists, prediction missing
        'True Negative'  -> both missing
    """
    
    if gt in [None, np.nan] and pred is None:
        return "True Negative"
    if gt in [None, np.nan] and pred is not None:
        return "False Positive"
    if gt is not None and pred is None:
        return "False Negative"
    if gt == pred:
        return "True"
    return "False"

combined_df['match'] = combined_df.apply(
    lambda row: evaluate_date_match(row['GT_Web'], row['clext']),
    axis=1
)

combined_df.to_csv('website_baseline_results.csv', index=False)

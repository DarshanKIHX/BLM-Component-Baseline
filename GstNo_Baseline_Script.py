import os
import pandas as pd
import json
import re
from datetime import datetime, timedelta
import dateutil
from tqdm import tqdm
import numpy as np
import os
import traceback
from get_metadata import clean_value_text 
from get_metadata import process_metadata_final
from get_metadata import get_metadata
import ast
import argparse

# Setup argument parser
parser = argparse.ArgumentParser(description='Process ground truth CSV file path.')
parser.add_argument('--gt_path', type=str, required=True, help='Path to the ground truth CSV file')
args = parser.parse_args()

# Load gt_df using the argument
gt_df = pd.read_csv(args.gt_path)
gt_df= gt_df[gt_df['status']=='completed']

def extract_comment(x):
    try:
        d = ast.literal_eval(x)  # Safely parse string to dict
        return d.get('comment') if isinstance(d, dict) else None
    except (ValueError, SyntaxError):
        return None  # Return None if parsing fails

gt_df['comment_value'] = gt_df['metadata'].apply(extract_comment)

rel_gt=gt_df[['claim_id','document_id','key_text','value_text','page_id','comment_value']]

blm_output_dir = '/home/nithilanv/workspace/BLM_Component_Baseline_Parakeet/blm_outputs'

blm_output_files = os.listdir(blm_output_dir)

blm_output_paths = [os.path.join(blm_output_dir, blm_filename) 
                    for blm_filename in blm_output_files]

pr2 = pd.DataFrame()
pr2.to_csv('blank_csv.csv')

pr2=pd.read_csv('/home/darshank/workspace/gst_no_extr/gst_no_baseline/blank_csv.csv')

mapper=pd.read_csv('/home/darshank/workspace/IHX_BLM_Consolidated_mapper.csv')

gno_df = []
error_dict = {}
for path in tqdm(blm_output_paths):

    try:
        txn_id = (path.split('_')[-3].split('/')[-1])
        page_id = (path.split('_')[-1].split('.')[0])
        if not txn_id in rel_gt['document_id'].tolist():
            #print('meow')
            continue
        gst_numbers = None

        with open(path, 'r') as file:
            data = json.load(file)
            
        meta = data.get('meta_data')
        if not meta:
            continue
        df = pd.DataFrame(meta)
        final_meta = process_metadata_final(get_metadata(df), mapper, pr2)

        gst_numbers = final_meta.loc[final_meta['standard_label'] == 'Gst_No', 'processed_ner'].tolist()
        picked_up_keys = final_meta.loc[final_meta['standard_label'] == 'Gst_No', 'ner_key'].tolist()

    except Exception as e:
        traceback.print_exc()
        print(f"Error processing {path}: {e}")
        error_dict[path] = e
        continue

    temp_dict = {
        'txn_id': txn_id,
        'page_id': page_id,
        'gst_numbers': gst_numbers,
        'picked_up_keys': picked_up_keys
    }

    gno_df.append(temp_dict)

gn_df= pd.DataFrame(gno_df)

gn_df.to_csv('extr_BLM_gstno.csv')

gt_df=pd.read_csv('/home/darshank/workspace/gst_no_extr/gst_no_baseline/parakeet_annotated_set1_gst_no.csv')

def extract_comment(x):
    try:
        d = ast.literal_eval(x)  # Safely parse string to dict
        return d.get('comment') if isinstance(d, dict) else None
    except (ValueError, SyntaxError):
        return None  # Return None if parsing fails

gt_df['comment_value'] = gt_df['metadata'].apply(extract_comment)

gt_df= gt_df[gt_df['status']=='completed'

gn_gt= gt_df[['document_id', 'page_id', 'value_text', 'claim_id','comment_value','status']]

gn_gt.rename(columns={'document_id': 'txn_id'}, inplace=True)
gn_gt.rename(columns={'value_text': 'GT_Gst_No'}, inplace=True)

gn_df['key'] = gn_df.apply(lambda row: row['txn_id'] + '_' + str(row['page_id']), axis = 1)

gn_gt['key'] = gn_gt.apply(lambda row: row['txn_id'] + '_' + str(row['page_id']), axis = 1)

combined_df = gn_gt.merge(gn_df, how='left', on='key')

combined_df['clext'] = combined_df['gst_numbers']

combined_df['clext'] = combined_df['clext'].astype(str).str.replace(r"[\[\]']", '', regex=True)

def clean_gt_receipt_no(s):
    if isinstance(s, str):
        if s.startswith(':'):
            s = s[1:]
        for char in [' ', '-', '/', '.']:
            s = s.replace(char, '')
    return s

combined_df['GT_Gst_No'] = combined_df['GT_Gst_No'].apply(clean_gt_receipt_no)
combined_df['clext'] = combined_df['clext'].apply(clean_gt_receipt_no)

combined_df['clext'] = combined_df['clext'].replace(['None','NA','', None,'nan'], np.nan)

combined_df['match'] = combined_df.apply(lambda row: row['GT_Gst_No'] == row['clext'], axis = 1)

def evaluate_Rno_match(gt, pred):
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

combined_df["Rno_match"] = combined_df.apply(
    lambda row: evaluate_Rno_match(row["GT_Gst_No"], row["clext"]),
    axis=1
)
combined_df.to_csv('GST_baseline_results.csv', index=False)

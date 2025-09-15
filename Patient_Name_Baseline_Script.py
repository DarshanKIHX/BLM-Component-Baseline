import os
import pandas as pd
import json
import re
from tqdm import tqdm
import numpy as np 
import os
import traceback
from get_metadata import get_metadata, process_metadata_final

gt_df = pd.read_csv("enter the path to your gt csv file here")

import ast

def extract_comment(x):
    try:
        d = ast.literal_eval(x)  # Safely parse string to dict
        return d.get('comment') if isinstance(d, dict) else None
    except (ValueError, SyntaxError):
        return None  # Return None if parsing fails

gt_df['comment_value'] = gt_df['metadata'].apply(extract_comment)

rel_gt_df = gt_df[['value_text', 'claim_id', 'document_id', 'page_id', 'comment_value']]

blm_output_dir = '/mnt/ihxaidata01/nithilanv/parakeet_annotated_data/blm_outputs'

blm_output_files = os.listdir(blm_output_dir)

blm_output_paths = [os.path.join(blm_output_dir, blm_filename) 
                    for blm_filename in blm_output_files]

patient_df = []
error_dict = {}

for path in tqdm(blm_output_paths):

    try:
        txn_id = (path.split('_')[-3].split('/')[-1])
        page_id = (path.split('_')[-1].split('.')[0])
        if txn_id not in rel_gt_df['document_id'].tolist():
            print('meow')
            continue

        with open(path, 'r') as file:
            data = json.load(file)

        mapper= pd.read_csv('/home/darshank/workspace/bill_date_extr/IHX_BLM_metadata_map_updated_04082025.csv')
        meta = data.get('meta_data')
        if not meta:
            continue
        df = pd.DataFrame(meta)
        final_meta = process_metadata_final(get_metadata(df), mapper)
        patient_names = final_meta.loc[final_meta['standard_label'] == 'Patient_Name', 'processed_ner'].tolist()


    except Exception as e:
        traceback.print_exc()
        print(f"Error processing {path}: {e}")
        error_dict[path] = e
        continue

    temp_dict = {
        'txn_id': txn_id,
        'page_id': page_id,
        'patient_names': patient_names
    }

    patient_df.append(temp_dict)

patient_name_blm_df= pd.DataFrame(patient_df)


pa_na_df= gt_df[['document_id', 'page_id', 'value_text', 'claim_id', 'comment_value']]
pa_na_df.rename(columns={'document_id': 'txn_id'}, inplace=True)
pa_na_df.rename(columns={'value_text': 'GT_Patient_Name'}, inplace=True)
pa_na_df.to_csv('extracted_Patient_Name_gt.csv', index=False)

patient_name_blm_df['key'] = patient_name_blm_df.apply(lambda row: row['txn_id'] + '_' + str(row['page_id']), axis = 1)

pa_na_df['key'] = pa_na_df.apply(lambda row: row['txn_id'] + '_' + str(row['page_id']), axis = 1)

combined_df = pa_na_df.merge(patient_name_blm_df, how='left', on='key')

combined_df.rename(columns={'value_text': 'GT_Patient_Name'}, inplace=True)
combined_df['Extr_Patient_Name'] = combined_df['patient_names']


combined_df['Extr_PN'] = combined_df['GT_Patient_Name'].apply(lambda x: x[0] if (isinstance(x, list) and len(x) > 0) else 'None')

def extract_patient_name(val):
    if isinstance(val, list) and len(val) > 0:
        return (None if val[0] == 'na' else val[0])
    return None

relevant_combined_df = combined_df[['txn_id', 'page_id', 'claim_id', 'GT_Patient_Name', 'Extr_Patient_Name','comment_value']]
relevant_combined_df['Extr_Patient_Name'] = relevant_combined_df['Extr_Patient_Name'].apply(extract_patient_name)

relevant_combined_df['match'] = relevant_combined_df.apply(lambda row: row['GT_Patient_Name'] == row['Extr_Patient_Name'], axis = 1)

relevant_combined_df['GT_Patient_Name'] = relevant_combined_df['GT_Patient_Name'].str.lstrip(': ')

from get_metadata import clean_value_text

relevant_combined_df['Extr_Patient_Name'] = relevant_combined_df['Extr_Patient_Name'].apply(lambda x: clean_value_text(x).lower() if not pd.isna(x) else x)



relevant_combined_df['GT_Patient_Name'] = relevant_combined_df['GT_Patient_Name'].str.replace(r'[^a-zA-Z0-9]', '', regex=True).str.lower()


relevant_combined_df['match'] = relevant_combined_df.apply(lambda row: row['GT_Patient_Name'] == row['Extr_Patient_Name'], axis = 1)

relevant_combined_df['match'] = relevant_combined_df.apply(lambda row: row['GT_Patient_Name'] == row['Extr_Patient_Name'], axis = 1)

def compare_patient_data(gt, extr, txn_id_y):
    if pd.isna(txn_id_y):
        return 'No BLM Output'
    elif pd.isna(gt) and pd.isna(extr):
        return 'True Negative'
    elif gt == extr:
        return 'Match'
    else:
        return 'No Match'

combined_df['Match_Result'] = combined_df.apply(
    lambda row: compare_patient_data(row['GT_Patient_Name'], row['Extr_Patient_Name'], row['txn_id_y']),
    axis=1
)


combined_df['Match_Result'].value_counts()

relevant_combined_df[(relevant_combined_df['GT_Patient_Name'].isna()) & (relevant_combined_df['Extr_Patient_Name'].isna())]['match'] = 'True Negative' 

def evaluate_PName_match(gt, pred):
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

relevant_combined_df['match'] = relevant_combined_df.apply(
    lambda row: evaluate_PName_match(row['GT_Patient_Name'], row['Extr_Patient_Name']),
    axis=1
)

relevant_combined_df.loc[relevant_combined_df['GT_Patient_Name']=='None', 'match'] = 'No BLM Output'


relevant_combined_df['match'].value_counts()
combined_df.to_csv('Patient_Name_baseline_results.csv', index=False)

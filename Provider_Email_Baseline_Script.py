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
from get_metadata import get_metadata


import ast


gt_df=pd.read_csv('/home/darshank/workspace/provider_email_extr/parakeet_annotated_set1_provider_email.csv')


rel_gt=gt_df[['claim_id','document_id','key_text','value_text','page_id','comment_value']]


blm_output_dir = '/home/nithilanv/workspace/BLM_Component_Baseline_Parakeet/blm_outputs'


blm_output_files = os.listdir(blm_output_dir)


blm_output_paths = [os.path.join(blm_output_dir, blm_filename) 
                    for blm_filename in blm_output_files]


pr2 = pd.DataFrame()


mapper=pd.read_csv('/home/darshank/workspace/provider_email_extr/pe_temp_mapper.csv')


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


pp_df = []
error_dict = {}
for path in tqdm(blm_output_paths):
    try:
        txn_id = (path.split('_')[-3].split('/')[-1])
        page_id = (path.split('_')[-1].split('.')[0])
        if not txn_id in rel_gt['document_id'].tolist():
            print('meow')
            continue
        prph = None

        with open(path, 'r') as file:
            data = json.load(file)
            
        meta = data.get('meta_data')
        if not meta:
            continue
        df = pd.DataFrame(meta)

        raw = pd.DataFrame(data.get('ocr+lm_raw_output', []))
        header = raw[raw['label'].str.contains('Header')].copy()

        header_objects = get_header_objects(header)

        #final_meta = process_metadata_final(get_metadata(df), mapper, pr2)

        #prph = final_meta.loc[final_meta['standard_label'] == 'Provider_Phone', 'processed_ner'].tolist()
        #picked_up_keys = final_meta.loc[final_meta['standard_label'] == 'Provider_Phone', 'ner_key'].tolist()

    except Exception as e:
        #traceback.print_exc()
        #print(f"Error processing {path}: {e}")
        error_dict[path] = e
        continue

    temp_dict = {
        'txn_id': txn_id,
        'page_id': page_id,
        #'provider phone': prph,
        #'picked_key': picked_up_keys,
        'header_objects': header_objects,
        'header_rows': header 
    }

    pp_df.append(temp_dict)


gn_df= pd.DataFrame(pp_df)


import re
import numpy as np

def extract_emails_from_list(text_list):
    if not isinstance(text_list, list):
        return np.nan
    
    combined_text = ' '.join(str(t) for t in text_list)
    
    # Step 1: fix spaces around '@' and '.'
    combined_text = re.sub(r'\s*@\s*', '@', combined_text)
    combined_text = re.sub(r'\s*\.\s*', '.', combined_text)
    
    # Step 2: remove ALL spaces (to fix broken emails)
    combined_text_no_spaces = combined_text.replace(' ', '')
    
    # Step 3: initial loose regex to grab emails + trailing chars (if any)
    # We'll grab the longest strings that look like emails plus some trailing letters
    email_candidate_pattern = r'[a-zA-Z0-9._\-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}[a-zA-Z0-9]*'
    candidates = re.findall(email_candidate_pattern, combined_text_no_spaces)
    
    cleaned_emails = []
    # Valid TLDs to stop trimming at, extend as needed
    valid_tlds = ['com', 'in', 'org', 'net', 'gov', 'edu', 'co', 'us', 'uk']
    
    for candidate in candidates:
        # Find last '.' to get TLD
        last_dot_index = candidate.rfind('.')
        if last_dot_index == -1:
            continue
        
        # Extract TLD candidate (up to 5 chars after last dot to cover things like .com, .co.uk, etc.)
        possible_tld = candidate[last_dot_index+1:last_dot_index+6].lower()
        
        # Trim trailing letters after valid TLD
        trimmed_email = candidate
        for tld in valid_tlds:
            if possible_tld.startswith(tld):
                # Trim after the TLD length + last_dot_index +1
                trim_point = last_dot_index + 1 + len(tld)
                trimmed_email = candidate[:trim_point]
                break
        
        cleaned_emails.append(trimmed_email)
    
    if cleaned_emails:
        # Remove duplicates
        unique_emails = list(set(cleaned_emails))
        return ', '.join(unique_emails)
    else:
        return np.nan


import re
import numpy as np

def extract_emails_from_list2(text_list):
    if not isinstance(text_list, list):
        return np.nan
    
    combined_text = ' '.join(str(t) for t in text_list)
    
    # Remove everything before and including the word "email" (case insensitive),
    # plus any special chars like : . , / ; right after it
    email_pos = re.search(r'email\s*[:.,/;]*', combined_text, flags=re.IGNORECASE)
    if email_pos:
        # Keep only the substring after that match
        combined_text = combined_text[email_pos.end():]
    
    # Step 1: fix spaces around '@' and '.'
    combined_text = re.sub(r'\s*@\s*', '@', combined_text)
    combined_text = re.sub(r'\s*\.\s*', '.', combined_text)
    
    # Step 2: remove ALL spaces (to fix broken emails)
    combined_text_no_spaces = combined_text.replace(' ', '')
    
    # Step 3: initial loose regex to grab emails + trailing chars (if any)
    email_candidate_pattern = r'[a-zA-Z0-9._\-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}[a-zA-Z0-9]*'
    candidates = re.findall(email_candidate_pattern, combined_text_no_spaces)
    
    cleaned_emails = []
    valid_tlds = ['com', 'in', 'org', 'net', 'gov', 'edu', 'co', 'us', 'uk','in']
    
    for candidate in candidates:
        # Remove 4 to 10 leading digits if present
        candidate = re.sub(r'^\d{4,10}', '', candidate)
        
        last_dot_index = candidate.rfind('.')
        if last_dot_index == -1:
            continue
        
        possible_tld = candidate[last_dot_index+1:last_dot_index+6].lower()
        
        trimmed_email = candidate
        for tld in valid_tlds:
            if possible_tld.startswith(tld):
                trim_point = last_dot_index + 1 + len(tld)
                trimmed_email = candidate[:trim_point]
                break
        
        cleaned_emails.append(trimmed_email)
    
    if cleaned_emails:
        unique_emails = list(set(cleaned_emails))
        return ', '.join(unique_emails)
    else:
        return np.nan



gn_df['extr_email'] = gn_df['header_objects'].apply(extract_emails_from_list)



gn_df['extr_email2'] = gn_df['header_objects'].apply(extract_emails_from_list2)


gn_df['clext'] = gn_df['extr_email']

extr_df = gn_df[['txn_id', 'page_id', 'extr_email2', 'clext','header_objects']]


def extract_comment(x):
    try:
        d = ast.literal_eval(x)  # Safely parse string to dict
        return d.get('comment') if isinstance(d, dict) else None
    except (ValueError, SyntaxError):
        return None  # Return None if parsing fails

gt_df['comment_value'] = gt_df['metadata'].apply(extract_comment)
gn_gt=gt_df

gn_gt.rename(columns={'document_id': 'txn_id'}, inplace=True)
gn_gt.rename(columns={'value_text': 'GT_PE'}, inplace=True)


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
    lambda row: evaluate_date_match(row['GT_PE'], row['clext']),
    axis=1
)

combined_df.to_csv('provider_email_results.csv', index=False)

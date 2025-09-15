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
from get_metadata import convert_to_standard_format
import ast
import argparse

# Setup argument parser
parser = argparse.ArgumentParser(description='Process ground truth CSV file path.')
parser.add_argument('--gt_path', type=str, required=True, help='Path to the ground truth CSV file')
args = parser.parse_args()

# Load gt_df using the argument
gt_df = pd.read_csv(args.gt_path)

rel_gt_df = gt_df[['value_text', 'claim_id', 'document_id', 'page_id', 'comment_value']]
#enter the blm output directory
blm_output_dir = '/mnt/ihxaidata01/nithilanv/parakeet_annotated_data/blm_outputs'

blm_output_files = os.listdir(blm_output_dir)
blm_output_paths = [os.path.join(blm_output_dir, blm_filename) 
                    for blm_filename in blm_output_files]
bd_df = []
error_dict = {}
for path in tqdm(blm_output_paths):
    try:
        txn_id = (path.split('_')[-3].split('/')[-1])
        page_id = (path.split('_')[-1].split('.')[0])

        if not txn_id in rel_gt_df['document_id'].tolist():
            print('bruh')
            continue
        age = None
        with open(path, 'r') as file:
            data = json.load(file)
            
        metadata = data.get('post_meta_data', [])
        final_meta = pd.DataFrame(metadata)

        if final_meta.empty:
            bill_dates = None
        else:            
            bill_dates = final_meta.loc[final_meta['standard_label'] == 'Bill_Date', 'ner_value'].tolist()
            bill_dates = [convert_to_standard_format(bill_date) for bill_date in bill_dates]

    except Exception as e:
        print(f"Error processing {path}: {e}")
        error_dict[path] = e
        continue  # continue on error instead of break
    
    temp_dict = {
        'txn_id': txn_id,
        'page_id': page_id,
        'bill_dates': bill_dates
    }
    bd_df.append(temp_dict)

Bill_Date_df = pd.DataFrame(bd_df)

def convert_to_standard_format(date_string):
    # Remove any leading/trailing whitespace and tabs
    if not isinstance(date_string, str) or not date_string.strip():
        return None
    date_string = date_string.replace("\t", " ").strip()  # Replace tabs with spaces and strip whitespace
    date_string = date_string.replace(",", " ").replace("..", ".")
    date_string = (
        re.sub(r"[^a-zA-Z0-9\s\-\/\.:]", "", date_string).lstrip("0").replace(" . ", ".")
    )
    date_string = re.sub(r'\s*\d{1}\s*(AM|PM)', '', date_string)
    date_string = " ".join(date_string.split()).lower()
    # print(f"Processed Date String: {date_string}")  # Debug line to see the processed date string
    # Dictionary to map month abbreviations to numbers
    month_map = {
        "jan": "01",
        "feb": "02",
        "mar": "03",
        "apr": "04",
        "may": "05",
        "jun": "06",
        "jul": "07",
        "aug": "08",
        "sep": "09",
        "oct": "10",
        "nov": "11",
        "dec": "12",
    }

    # Define regex patterns for all formats
    dd_mm_yyyy_pattern = r"(\d{1,2})[-/. ](\d{1,2})[-/. ](\d{4}):*"  # e.g., 20/12/2024
    dd_mm_yy_pattern = r"(\d{1,2})[-/. ](\d{1,2})[-/. ](\d{2})"    # e.g., 20/12/24
    yyyy_mm_dd_pattern = r"(\d{4})[-/. ](\d{1,2})[-/. ](\d{1,2})"
    dd_mon_yyyy_pattern = r"(\d{1,2})[-/. ]([a-zA-Z]{3,12})[-/. ](\d{4})"
    dd_mon_yy_pattern = r"(\d{1,2})[-/. ]([a-zA-Z]{3,12})[-/. ](\d{2})"
    yyyy_mm_dd_time_tz_pattern = r"\d{4}-\d{2}-\d{2}\s*\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}"
    mon_dd_yyyy_pattern = r"([a-zA-Z]{3,12})[-/. ](\d{1,2})[-/. ](\d{4})"
    yyyy_mon_dd_pattern = r"(\d{4})[-/. ]([a-zA-Z]{3,12})[-/. ](\d{2})"
    ddmmyyyy_pattern = r"(\d{2})(\d{2})(\d{4})"
    spaceslash_ddmmyyyy_pattern = r"(\d{1,2})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*(\d{4})\s*"
    dd_mm_yy_time_pattern = r"(\d{2})/(\d{2})/(\d{2})\s*(\d{2}):(\d{2})"
    # dd_mm_yyyy_time_pattern = r"(\d{2})/(\d{2})/(\d{4})\s*(\d{1,2})\s*[.:]?\s*(\d{2})\s*(AM|PM)?"
    # dd_mm_yy_am_pm_pattern = r"(\d{2})/(\d{2})/(\d{2})\s*(\d{1,2}):(\d{2})\s*(AM|PM)"

    date_obj = None
    try:
        # Try different formats using regex and handle accordingly

        if re.search(dd_mm_yyyy_pattern, date_string) != None:
            # Indian format: dd-mm-yyyy
            day, month, year = re.findall(dd_mm_yyyy_pattern, date_string)[0]
            date_obj = datetime(int(year), int(month), int(day))

        elif re.search(dd_mm_yy_time_pattern, date_string):
            # Format: dd/mm/yy hh:mm (e.g., 20/12/24 11:19)
            day, month, year, hour, minute = re.findall(dd_mm_yy_time_pattern, date_string)[0]
            year = "20" + year  # Convert 2-digit year to 4-digit year
            date_obj = datetime(int(year), int(month), int(day), int(hour), int(minute))
        
        elif re.search(yyyy_mm_dd_pattern, date_string) != None:
            # ISO format: yyyy-mm-dd
            year, month, day = re.findall(yyyy_mm_dd_pattern, date_string)[0]
            date_obj = datetime(int(year), int(month), int(day))
           
        
        elif re.search(yyyy_mm_dd_time_tz_pattern, date_string) != None:
            # Format: 2024-10-09 12:00:00+00:00
            date_obj = dateutil.parser.parse(date_string)
        
        elif re.search(yyyy_mon_dd_pattern, date_string) != None:
            # Format: yyyy-mon-dd
            year, mon, day = re.findall(yyyy_mon_dd_pattern, date_string)[0]
            month = month_map[mon.lower()[:3]]
            date_obj = datetime(int(year), int(month), int(day))
        
        elif (
            re.search(dd_mm_yy_pattern, date_string) != None
            and re.search(yyyy_mm_dd_pattern, date_string) == None
        ):
            # Format: dd-mm-yy (two-digit year)
            day, month, year = re.findall(dd_mm_yy_pattern, date_string)[0]
            year = "20" + year
            date_obj = datetime(int(year), int(month), int(day))

        elif re.search(dd_mon_yyyy_pattern, date_string) != None:
            # Format: dd-mon-yyyy
            day, mon, year = re.findall(dd_mon_yyyy_pattern, date_string)[0]
            month = month_map[mon.lower()[:3]]
            date_obj = datetime(int(year), int(month), int(day))

        elif re.search(dd_mon_yy_pattern, date_string) != None:
            # Format: dd-mon-yy
            day, mon, year = re.findall(dd_mon_yy_pattern, date_string)[0]
            month = month_map[mon.lower()[:3]]
            year = "20" + year  # Assuming all years are in the 21st century
            date_obj = datetime(int(year), int(month), int(day))


        elif re.search(mon_dd_yyyy_pattern, date_string) != None:
            # Format: mon-dd-yyyy
            mon, day, year = re.findall(mon_dd_yyyy_pattern, date_string)[0]
            month = month_map[mon.lower()[:3]]
            date_obj = datetime(int(year), int(month), int(day))
        
        elif re.search(spaceslash_ddmmyyyy_pattern, date_string) != None:
            # Format: dd/mm/yyyy with space
            day, month, year = re.findall(spaceslash_ddmmyyyy_pattern, date_string)[0]
            date_obj = datetime(int(year), int(month), int(day))

        elif re.search(ddmmyyyy_pattern, date_string) != None:
            # Format: ddmmyyyy
            day, month, year = re.findall(ddmmyyyy_pattern, date_string)[0]
            date_obj = datetime(int(year), int(month), int(day))
        
        else:
            raise ValueError("Invalid date format")

        # Format the date in the desired output format
        if (
            date_obj
            and date_obj > (datetime.now() - timedelta(days=365))
            and (date_obj < datetime.now() + timedelta(days=1))
        ):
            return date_obj.strftime("%Y-%m-%dT12:00:00.000Z")
        else:
            return None

    except (ValueError, IndexError, KeyError) as e:
        # If parsing fails, return None or original string
       return None  # or return date_string if you prefer
    
def extract_comment(x):
    try:
        d = ast.literal_eval(x)  # Safely parse string to dict
        return d.get('comment') if isinstance(d, dict) else None
    except (ValueError, SyntaxError):
        return None  # Return None if parsing fails

gt_df['comment_value'] = gt_df['metadata'].apply(extract_comment)

gt_df["standardized_date"] =gt_df.apply(
    lambda row: convert_to_standard_format(row["comment_value"])
                if pd.notna(row["comment_value"]) and row["comment_value"] is not None
                else (
                    convert_to_standard_format(row["value_text"])
                    if pd.notna(row["value_text"]) and row["value_text"] is not None
                    else np.nan
                ),
    axis=1
)

idk_df= gt_df[['document_id', 'page_id', 'standardized_date', 'claim_id']]

idk_df.rename(columns={'document_id': 'txn_id'}, inplace=True)

idk_df.to_csv('extracted_bill_date_gt.csv', index=False)

Bill_Date_df['key'] = Bill_Date_df.apply(lambda row: row['txn_id'] + '_' + str(row['page_id']), axis = 1)
idk_df['key'] = idk_df.apply(lambda row: row['txn_id'] + '_' + str(row['page_id']), axis = 1)
combined_df = idk_df.merge(Bill_Date_df, how='left', on='key')
combined_df.rename(columns={'standardized_date': 'GT_Bill_Date'}, inplace=True)
combined_df.rename(columns={'Bill_Date': 'Extr_Bill_Date'}, inplace=True)
combined_df['Extr_BD'] = combined_df['GT_Bill_Date'].apply(lambda x: x[0] if (isinstance(x, list) and len(x) > 0) else 'None')
relevant_combined_df=combined_df
relevant_combined_df['GT_Bill_Date'].fillna('None', inplace=True)
def extract_extr_bill_date(val):
    if isinstance(val, list) and len(val) > 0:
        return None if val[0]=='na' else val[0]
        return 'None'

relevant_combined_df['bill_date']= relevant_combined_df['bill_dates'].apply(extract_extr_bill_date)
relevant_combined_df['match'] = relevant_combined_df.apply(lambda row: row['GT_Bill_Date'] == row['bill_date'], axis = 1)
relevant_combined_df['match'] = relevant_combined_df.apply(lambda row: row['GT_Bill_Date'] == row['bill_date'], axis = 1)

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

relevant_combined_df['match'] = relevant_combined_df.apply(
    lambda row: evaluate_date_match(row['GT_Bill_Date'], row['bill_date']),
    axis=1
)
relevant_combined_df.loc[relevant_combined_df['GT_Bill_Date']=='None', 'match'] = 'No BLM Output'
relevant_combined_df["match"].value_counts()
relevant_combined_df.to_csv('bill_date_evaluation.csv', index=False)

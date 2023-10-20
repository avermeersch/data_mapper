import pandas as pd
import os
import openai
import json
from pandas.api.types import is_datetime64_any_dtype
from retry import retry

def infer_data_types(df):
    """
    Infer data types of each column. This is optional and can be used for later stages.
    
    Parameters:
        df (pd.DataFrame): DataFrame with columns to infer data types.
        
    Returns:
        dict: A dictionary mapping columns to their inferred data types.
    """
    return {col: str(df[col].dtype) for col in df.columns}

def extract_column_information(df, num_samples=3):
    """
    Extract column information along with a few samples.

    Parameters:
        df (pd.DataFrame): DataFrame for which column information is to be extracted.
        num_samples (int): Number of sample values to include for each column.

    Returns:
        str: Description of columns including data type and samples.
    """
    
    columns = df.columns.tolist()
    data_types = {col: str(df[col].dtype) for col in columns}
    description = ""

    for col in columns:
        samples = df[col].dropna().sample(min(num_samples, len(df))).tolist()
        sample_str = ", ".join([str(s) for s in samples])
        description += f"The column '{col}' is of type {data_types[col]} with examples: {sample_str}.\n"

    return description

@retry(tries=60,delay=1)
def get_openai_response(prompt):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant specializing in finding mappings between columns of SQL queries as well as the associated mappings between data formatting. "},
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response['choices'][0]['message']['content']

def find_similar_columns(template_table, candidate_tables):
    # Setup the prompt template
    template = """
    Question: For the column '{column_candidate}' from the candidate table, example values {candidate_example_values}, determine if it matches or closely resembles any column from the template table with columns {columns_template}.
    The problem here is that the columns can be combined together into another table but due to column naming or data naming conventions, the columns can be slightly different. 
    For instance, a column called "Cost" with numbers in it, and a column called "Monthly Cost" with numbers in it, should return that they match. 
    Likewise, if there's a date column called "Date" and another date column called "Start Date" then this could also match. 
    
    Evaluation Criteria:
    1. Datatype: Consider if the datatype of '{column_candidate}' is compatible with the datatypes of columns in {columns_template}.
    2. Naming Convention: Factor in semantic similarity and naming variations (e.g., 'date' vs 'PlanDate').
    3. Value Patterns: Take into account the formats or categories of the values (e.g., different datetime formats, 'Gold Plan' vs 'Gold').
    4. Ordinality: If the column contains ordinal values, consider the order of categories.
    5. Units: Be mindful of unit differences (e.g., meters vs feet, dollars vs euros).
    6. Language: Consider language-specific characteristics, such as special characters or diacritics.

    Special Edge Cases (Include in the 'Ambiguous' list if applicable):
    - Partial Match due to unit differences.
    - Partial Match due to significant missing values.
    - Partial Match due to language-specific characteristics.

    Here is how I want you to respond. Your response should only be one of the following options:
    - Return 'Match' and mention the exact matching column name if the column '{column_candidate}' directly matches or is very similar to a specific column in {columns_template} according to the evaluation criteria.
    - Return 'No Match' if the column '{column_candidate}' does not match or resemble any columns in {columns_template} based on the evaluation criteria.
    - Return 'Ambiguous' and list the potential matching columns if there is uncertainty or the column '{column_candidate}' could potentially match multiple columns in {columns_template} based on the evaluation criteria.

    Your answer MUST BE either: 'Match', 'No Match', or 'Ambiguous'. 
    
    Answer:"""

    # Extract columns from the template table
    template_columns = template_table.columns.tolist()

    # Dictionary to hold results
    results = {}

    # For each candidate table, compare columns with the template
    for candidate_name, candidate_table in candidate_tables.items():
        candidate_columns = candidate_table.columns.tolist()
        column_matches = {}
        column_metadata = {}

        for column_candidate in candidate_columns:
            candidate_example_values = candidate_table[column_candidate].dropna().sample(min(3, len(candidate_table[column_candidate]))).tolist()

            metadata = {
                'example_values': candidate_example_values,
            }

            # Store the metadata for each column
            column_metadata[column_candidate] = metadata

            prompt = template.format(
                column_candidate=column_candidate, 
                candidate_example_values=candidate_example_values,
                columns_template=template_columns
            )
            
            response = get_openai_response(prompt)
            print(response)

            # Handle ambiguous mapping by prompting user
            if "Ambiguous" in response:
                print(f"Ambiguous mapping detected for column '{column_candidate}' in {candidate_name}.")
                print("Potential matches from template table are:")
                for idx, potential_match in enumerate(template_columns, 1):
                    print(f"{idx}. {potential_match}")
                choice = int(input(f"Please select the most suitable match for '{column_candidate}': "))
                chosen_match = template_columns[choice-1]
                column_matches[column_candidate] = chosen_match
            else:
                column_matches[column_candidate] = response
        
        results[candidate_name] = {'matches': column_matches, 'metadata': column_metadata}

    return results

def generate_transformation_code(column_mapping, metadata):
    """
    Generate Python code to transform data from the source table to the target table.
    
    Parameters:
        column_mapping (dict): Mapping from source column names to target column names.
        metadata (dict): Metadata about source columns.
        
    Returns:
        str: Python code to perform the transformations.
    """
    code_list = []
    
    for source_col, target_col in column_mapping.items():
        if target_col == 'No Match':
            continue
            
        src_dtype = metadata[source_col]['datatype']
        src_format = metadata[source_col]['format']  # Assuming format is provided in metadata
        target_format = metadata[target_col]['format']  # Assuming format is also provided for target columns

        if pd.api.types.is_integer_dtype(src_dtype):
            code_list.append(f"# Map integer column '{source_col}' to '{target_col}'")
            code_list.append(f"df['{target_col}'] = df['{source_col}'].astype(int)\n")

        elif pd.api.types.is_float_dtype(src_dtype):
            code_list.append(f"# Map float column '{source_col}' to '{target_col}'")
            code_list.append(f"df['{target_col}'] = df['{source_col}'].astype(float)\n")

        elif pd.api.types.is_string_dtype(src_dtype):
            code_list.append(f"# Map string column '{source_col}' to '{target_col}'")
            code_list.append(f"df['{target_col}'] = df['{source_col}'].astype(str)\n")

        elif pd.api.types.is_datetime64_any_dtype(src_dtype):
            code_list.append(f"# Convert datetime column '{source_col}' to the format of '{target_col}'")
            if src_format != target_format:
                code_list.append(f"df['{target_col}'] = pd.to_datetime(df['{source_col}']).dt.strftime('{target_format}')\n")
            else:
                code_list.append(f"df['{target_col}'] = df['{source_col}']\n")

        else:
            # General code for column mapping, if not any of the above types
            code_list.append(f"# Code to map '{source_col}' to '{target_col}'")
            code_list.append(f"df['{target_col}'] = df['{source_col}']\n")
    
    return '\n'.join(code_list)

def check_data_transfer(template_df, transformed_df):
    """
    Check that all data has been transferred correctly.
    
    Parameters:
        template_df (pd.DataFrame): DataFrame of the template table.
        transformed_df (pd.DataFrame): DataFrame of the transformed source table.
        
    Returns:
        list: List of alerts regarding incorrect data transfers, empty if everything is okay.
    """
    alerts = []
    for col in template_df.columns:
        if col not in transformed_df.columns:
            alerts.append(f"Alert: Column '{col}' from template not found in transformed DataFrame.")
            
        elif not is_datetime64_any_dtype(template_df[col]) and not template_df[col].equals(transformed_df[col]):
            alerts.append(f"Alert: Column '{col}' has discrepancies between the template and transformed DataFrame.")
            
    return alerts
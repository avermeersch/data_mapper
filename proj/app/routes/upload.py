import sys
import json
from flask import Blueprint, request, jsonify, render_template

sys.path.append('/home/avermeer/proj/app/scripts')

import app.scripts.data_extraction as fe
import app.scripts.column_mapping as cm

upload_blueprint = Blueprint('upload', __name__)

@upload_blueprint.route('/upload', methods=['POST'])
def upload():
    # Handle file absence
    for expected_file_key in ['file', 'template', 'table_a', 'table_b']:
        if expected_file_key not in request.files:
            return jsonify({'error': f"No {expected_file_key} provided"}), 400

    template_file = request.files['template']
    table_a_file = request.files['table_a']
    table_b_file = request.files['table_b']

    # 1. Load and normalize data
    template_df = fe.load_data(template_file)
    table_a_df = fe.load_data(table_a_file)
    table_b_df = fe.load_data(table_b_file)

    template_df = fe.normalize_columns(template_df)
    table_a_df = fe.normalize_columns(table_a_df)
    table_b_df = fe.normalize_columns(table_b_df)

    # 2. Extract column information
    template_desc = cm.extract_column_information(template_df)
    table_a_desc = cm.extract_column_information(table_a_df)
    table_b_desc = cm.extract_column_information(table_b_df)

    tables = {
        "Table A": table_a_df,
        "Table B": table_b_df
    }

    results = cm.find_similar_columns(template_df, tables)

    # Convert the results to a JSON string and save to a JSON file
    results_json_str = json.dumps(results, indent=4)
    with open("results.json", "w") as f:
        f.write(results_json_str)

    # 3. Generate mapping code (This was commented out in your main)
    # Transformation code can be generated similarly, just as you commented out in main

    # 4. Check data transfer correctness and generate alerts
    alerts = {}
    for table_name, table_df in tables.items():
        transformed_df = table_df  # Placeholder, as in your main
        alerts[table_name] = cm.check_data_transfer(template_df, transformed_df)

    # Convert alerts to JSON string and save
    alerts_json_str = json.dumps(alerts, indent=4)
    with open("alerts.json", "w") as f:
        f.write(alerts_json_str)

    return jsonify({
        'columns': data.columns.tolist(),
        'results': results,
        'alerts': alerts
    })

@upload_blueprint.route('/', methods=['GET'])
def root():
    return render_template('index.html')

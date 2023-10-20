@app.route('/suggest_mapping', methods=['POST'])
def suggest_mapping():
    template_columns = request.json.get('template_columns')
    table_columns = request.json.get('table_columns')

    if not template_columns or not table_columns:
        return jsonify({'error': 'Missing columns data'}), 400

    mappings = {col: fe.suggest_column_mapping(col, table_columns) for col in template_columns}
    
    return jsonify(mappings)

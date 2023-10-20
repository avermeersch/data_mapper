@app.route('/transform_date', methods=['POST'])
def transform_date():
    date_str = request.json.get('date')
    
    if not date_str:
        return jsonify({'error': 'Date string missing'}), 400
    
    transformed_date = fe.transform_date_format(date_str)
    
    return jsonify({'transformed_date': transformed_date})

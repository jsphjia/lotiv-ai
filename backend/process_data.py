from flask import request, jsonify

def process(coordinates):
    # Get user inputs from the request
    data = request.json
    floors = data.get('floors')
    bedrooms = data.get('bedrooms')
    bathrooms = data.get('bathrooms')

    # Store the data in variables
    result = {
        'floors': floors,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'normalized_coordinates': coordinates
    }

    return jsonify(result)
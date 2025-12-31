from flask import Flask, request, jsonify
from flask_cors import CORS
from generate_land_plot import plot_coordinates
from process_data import process, process

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variable to store bounding box data
bounding_box_data = None
plan_limits = None

@app.route('/bounding-box', methods=['POST'])
def receive_bounding_box():
    global bounding_box_data
    data = request.get_json()
    if data:
        if "rings" in data:  # Check if the shape contains rings (polygon data)
            for ring in data["rings"]:
                bounding_box_data = plot_coordinates(ring)  # Plot the coordinates immediately
        return jsonify({"message": "Shape data received and plotted successfully."}), 200
    else:
        return jsonify({"message": "No data received."}), 400

@app.route('/process-data', methods=['POST'])
def process_data():
    global bounding_box_data
    global plan_limits
    if bounding_box_data:
        plan_limits = process(bounding_box_data)
        return jsonify(plan_limits), 200
    else:
        return jsonify({"message": "No bounding box data available."}), 400

if __name__ == '__main__':
    app.run(debug=True)
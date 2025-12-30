from flask import Flask, request, jsonify
from flask_cors import CORS
from generate_land_plot import plot_coordinates

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/bounding-box', methods=['POST'])
def receive_bounding_box():
    data = request.get_json()
    if data:
        print("Received Shape Data:", data, flush=True)  # Print all shape data
        if "rings" in data:  # Check if the shape contains rings (polygon data)
            for ring in data["rings"]:
                # print("Ring Coordinates:", ring, flush=True)
                plot_coordinates(ring)  # Plot the coordinates immediately
        return jsonify({"message": "Shape data received and plotted successfully."}), 200
    else:
        return jsonify({"message": "No data received."}), 400

if __name__ == '__main__':
    app.run(debug=True)
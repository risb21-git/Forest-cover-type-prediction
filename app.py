from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('forest_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

# Forest type metadata
FOREST_TYPES = {
    1: {
        'name'  : 'Spruce / Fir',
        'badge' : 'Subalpine · Type 1',
        'desc'  : 'Dense coniferous stands found at high elevations above 2800m. Engelmann Spruce and Subalpine Fir dominate these cold, moist slopes where snow persists well into summer.',
        'elev'  : '2800m – 3500m',
        'clim'  : 'Cold & moist',
        'img'   : 'https://images.unsplash.com/photo-1448375240586-882707db888b?w=700&q=80',
        'emoji' : '🌲'
    },
    2: {
        'name'  : 'Lodgepole Pine',
        'badge' : 'Montane · Type 2',
        'desc'  : 'The most widespread forest type in the Rocky Mountains. Lodgepole Pine colonizes disturbed areas rapidly and forms dense, even-aged stands across a broad elevation range.',
        'elev'  : '2400m – 3000m',
        'clim'  : 'Subalpine cool',
        'img'   : 'https://images.unsplash.com/photo-1542273917363-3b1817f69a2d?w=700&q=80',
        'emoji' : '🌳'
    },
    3: {
        'name'  : 'Ponderosa Pine',
        'badge' : 'Lower Montane · Type 3',
        'desc'  : 'Open, park-like forests with a grassy understory. Ponderosa Pine thrives on dry, south-facing slopes at lower elevations where fire historically maintained its structure.',
        'elev'  : '2000m – 2500m',
        'clim'  : 'Warm & dry',
        'img'   : 'https://images.unsplash.com/photo-1509316785289-025f5b846b35?w=700&q=80',
        'emoji' : '🌵'
    },
    4: {
        'name'  : 'Cottonwood / Willow',
        'badge' : 'Riparian · Type 4',
        'desc'  : 'Lush riparian galleries hugging stream corridors. Cottonwood and Willow require constant soil moisture and form narrow but biologically rich linear forests along waterways.',
        'elev'  : '1800m – 2400m',
        'clim'  : 'Riparian moist',
        'img'   : 'https://images.unsplash.com/photo-1500534314209-a25ddb2bd429?w=700&q=80',
        'emoji' : '🍃'
    },
    5: {
        'name'  : 'Aspen',
        'badge' : 'Transitional · Type 5',
        'desc'  : 'Aspen groves form vast clonal colonies — each stand is genetically a single organism. They colonize post-fire areas and create brilliant golden displays in autumn.',
        'elev'  : '2200m – 2900m',
        'clim'  : 'Cool temperate',
        'img'   : 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=700&q=80',
        'emoji' : '🍂'
    },
    6: {
        'name'  : 'Douglas-fir',
        'badge' : 'Mesic Montane · Type 6',
        'desc'  : 'Douglas-fir occupies moist, north-facing slopes at mid-elevations. One of the tallest and most commercially important conifers in North America, it can live over 1000 years.',
        'elev'  : '2100m – 2800m',
        'clim'  : 'Cool & mesic',
        'img'   : 'https://images.unsplash.com/photo-1511497584788-876760111969?w=700&q=80',
        'emoji' : '🌿'
    },
    7: {
        'name'  : 'Krummholz',
        'badge' : 'Alpine Treeline · Type 7',
        'desc'  : 'At the edge of survival — Krummholz trees are sculpted by extreme wind, cold, and ice into prostrate, flag-form shapes. This is the boundary between forest and alpine tundra.',
        'elev'  : '3400m – 3800m',
        'clim'  : 'Alpine severe',
        'img'   : 'https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=700&q=80',
        'emoji' : '🏔️'
    }
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data     = request.get_json()
        features = np.array(data['features']).reshape(1, -1)

        prediction = int(model.predict(features)[0])
        confidence = round(float(model.predict_proba(features).max()) * 100, 2)
        info       = FOREST_TYPES.get(prediction, FOREST_TYPES[2])

        return jsonify({
            'forest_type' : prediction,
            'forest_name' : info['name'],
            'badge'       : info['badge'],
            'description' : info['desc'],
            'elevation'   : info['elev'],
            'climate'     : info['clim'],
            'image_url'   : info['img'],
            'emoji'       : info['emoji'],
            'confidence'  : confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

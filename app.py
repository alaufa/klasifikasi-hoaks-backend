from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from preprocess import preprocess_text
import os

app = Flask(__name__)
CORS(app)

vectorizer = joblib.load('vectorizer.pkl')
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Data diterima:", data)
        text = data.get('text', '')
        processed = preprocess_text(text)
        tfidf = vectorizer.transform([processed])
        pred = model.predict(tfidf)[0]
        label = 'Hoaks' if pred == 1 else 'Valid'
        return jsonify({"label": label})
    except Exception as e:
        print("Terjadi error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

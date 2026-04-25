from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, jsonify

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Create API
app = Flask(__name__)

@app.route('/')
def home():
    return "Iris Model Running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    prediction = model.predict([data])
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import MarianMTModel, MarianTokenizer


app = Flask(__name__)
CORS(app)

# Load pre-trained model and tokenizer
model_name = "jamm55/autotrain-improved-pidgin-model-2837583189"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

@app.post('/translate')
def translate_to_pidgin():

    text = request.json.get('text')
    
    
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Perform translation
    outputs = model.generate(**inputs)

    # Decode translated text
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify(translated_text)

@app.get('/')
def index():
    return jsonify({'message': 'Welcome to the translation API'})


if(__name__ == "__main__"):
    app.run(debug=True, host='0.0.0.0', port=5000)

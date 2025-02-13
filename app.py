from flask import Flask, render_template, jsonify, request
from transformers import BertTokenizer, BertForSequenceClassification
import os
import json
import torch


app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the saved model and tokenizer from Google Drive
model_path = "./BioBERT-Symptom-Disease"  # Replace with your model path in Drive
model = BertForSequenceClassification.from_pretrained(model_path)
model.to(device)

tokenizer = BertTokenizer.from_pretrained(model_path)


# adding routes
@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    # Using request.form.get to safely access the input
    msg = request.form.get("msg", "")
    if not msg:
        return jsonify({"error": "No input provided"}), 400

    input_symptoms = msg
    print(f"User Input: {input_symptoms}")

    # Get disease prediction
    predicted_disease = predict_disease(input_symptoms)
    
    print(f"Predicted Disease: {predicted_disease}")
    return jsonify({"predicted_disease": predicted_disease})




def predict_disease(symptoms):
    inputs = tokenizer(symptoms, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    # Move the input tensors to the same device as the model
    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = model(**inputs)
    
    # Get the predicted label (disease ID)
    predicted_label = torch.argmax(outputs.logits, dim=1).item()
    with open("./mapping.json", "r") as file:
        disease_mapping = json.load(file)

    reverse_disease_mapping = {v: k for k, v in disease_mapping.items()}
    predicted_disease = reverse_disease_mapping.get(predicted_label, "Unknown Disease")

    return predicted_disease



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
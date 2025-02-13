<<<<<<< HEAD
# end-end-medical
=======

# end-end-medical

#overview

This is a medical chatbot powered by a fine-tuned BioBERT model for symptom-based disease prediction. The chatbot allows users to input their symptoms, and it returns a predicted disease name based on the trained model. The model is fine-tuned using the dataset.

#Features

User-friendly chat interface built with HTML, CSS, and JavaScript.

Flask backend to handle requests and process user inputs.

BioBERT-based disease prediction model trained on symptom-to-disease mapping.

#Installation & Setup
1. Clone the Repository
https://github.com/MdTohidReza/end-end-medical

2. Install Dependencies

3. Download the Pretrained Model

Ensure that the trained BioBERT model is stored in Google Drive or a local directory. If using Google Drive, mount it in Google Colab:
from google.colab import drive
drive.mount('/content/drive')

Modify the path in app.py accordingly:
model = BertForSequenceClassification.from_pretrained("/content/drive/MyDrive/BioBERT-Symptom-Disease")
tokenizer = BertTokenizer.from_pretrained("/content/drive/MyDrive/BioBERT-Symptom-Disease")


4. Run the Flask App
python app.py

The chatbot will be accessible at localhost:8080

#Usage

Enter symptoms in the chatbot.
The model processes the input and predicts a possible disease.
The chatbot displays the predicted disease.

>>>>>>> 154ef63 (medicalChatBot)

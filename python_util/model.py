# Imports
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from transformers import pipeline

# Creates tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("d4data/bias-detection-model")
model = TFAutoModelForSequenceClassification.from_pretrained("d4data/bias-detection-model")

# Function to return classifier model for bias
def get_classifier():
    return pipeline('text-classification', model=model, tokenizer=tokenizer) # cuda = 0,1 based on gpu availability

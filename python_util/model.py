# Imports
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from transformers import pipeline

# Creates tokenizer and model
tokenizer_bias = AutoTokenizer.from_pretrained("d4data/bias-detection-model")
model_bias = TFAutoModelForSequenceClassification.from_pretrained("d4data/bias-detection-model")

tokenizer_tone = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model_tone = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")



# Function to return classifier model for bias
def get_classifier():
    return pipeline('text-classification', model=model_bias, tokenizer=tokenizer_bias) # cuda = 0,1 based on gpu availability

def get_tone_analysis():
    return pipeline('text-classification', model=model_tone, tokenizer=tokenizer_tone)
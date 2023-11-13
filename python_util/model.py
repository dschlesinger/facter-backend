# Imports
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from transformers import pipeline

# Creates tokenizer and model
tokenizer_bias = AutoTokenizer.from_pretrained("d4data/bias-detection-model")
model_bias = TFAutoModelForSequenceClassification.from_pretrained("d4data/bias-detection-model")
tokenizer_tone = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model_tone = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Initializes pipelines
bias_model = pipeline('text-classification', model=model_bias, tokenizer=tokenizer_bias)
tone_model = pipeline('text-classification', model=model_tone, tokenizer=tokenizer_tone)


def get_bias(text):
    '''
    Input: text
    Ouptut: bias score 
    '''
    pred = bias_model(text)
    print(pred)
    if pred[0]["label"] == "Biased":
        return -pred[0]["score"]
    else:
        return pred[0]["score"]

def get_tone(text):
    '''
    Input: text
    Ouptut: tone score (based on sentiment analysis)
    '''
    pred = tone_model(text)
    if pred[0]["label"] == "NEGATIVE":
        return -pred[0]["score"]
    else:
        return pred[0]["score"]
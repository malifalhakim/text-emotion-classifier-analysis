from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers_interpret import SequenceClassificationExplainer

model = AutoModelForSequenceClassification.from_pretrained("indobertweet-fine-tuned")
tokenizer = AutoTokenizer.from_pretrained("indolem/indobertweet-base-uncased")
classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)

def classify(text):
    text = text.strip().lower()
    result = classifier(text)
    yhat = result[0]['label']
    return yhat

def analyze_text(text):
    cls_explainer = SequenceClassificationExplainer(model, tokenizer)
    attributions = cls_explainer(text)
    return cls_explainer.visualize()


    
from bertopic import BERTopic
import os
from umap import UMAP
from bertopic.representation import KeyBERTInspired

# Prevent Stochastic Behavior (Mencegah Hasil Berubah-ubah setiap Menjalankan)
umap_model = UMAP(random_state = 42)

# Model representasi untuk topik
representation_model = KeyBERTInspired()

# Define topic model
topic_model = BERTopic(
    umap_model=umap_model,
    language="indonesian", 
    representation_model=representation_model,
    verbose = True, 
    nr_topics="auto",
    min_topic_size = 3
)

def get_topic(df):
    docs = df['text']
    topics, probs = topic_model.fit_transform(docs)
    return topic_model.visualize_barchart()
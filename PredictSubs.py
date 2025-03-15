import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import random

# load data
def load_data(csv_path):
    df = pd.read_csv(csv_path, encoding='utf-8', sep=';')
    return df

def load_reference_list(reference_csv_path):
    df = pd.read_csv(reference_csv_path, encoding='utf-8', sep=';')
    reference_list = df["Substanz"].str.strip().unique().tolist()
    return reference_list

# make labeled train data
def prepare_train_examples(df, add_negative_samples=False):
    train_examples = []
    for _, row in df.iterrows():
        train_examples.append(InputExample(texts=[row["input_text"], row["label"]], label=1.0)) 
    
    if add_negative_samples:
        unique_labels = df["label"].unique().tolist()
        for _, row in df.iterrows():
            negative_label = random.choice([lbl for lbl in unique_labels if lbl != row["label"]])
            train_examples.append(InputExample(texts=[row["input_text"], negative_label], label=0.0)) 
    
    return train_examples

# train model, alternative use ClinicalBioBERT
def train_model(train_examples, val_examples, model_name="multi-qa-mpnet-base-cos-v1", epochs=3):
    model = SentenceTransformer(model_name)
    train_loss = losses.CosineSimilarityLoss(model) #try different loss function
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    
    # Create evaluator for validation
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(val_examples, name="val-eval")
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator, 
        epochs=epochs,
        evaluation_steps=100,
        output_path="substance_extractor_model",
    )
    return model

#predict substances
def predict_substance(input_text, model, reference_list):
    corpus_embeddings = model.encode(reference_list, show_progress_bar=True, convert_to_tensor=True)
    query_embedding = model.encode(input_text, convert_to_tensor=True)
    
    results = util.semantic_search(query_embedding, corpus_embeddings, top_k=1)
    best_match = reference_list[results[0][0]["corpus_id"]]
    similarity_score = round(results[0][0]["score"], 2)
    
    return best_match, similarity_score

def encode_reference_list(model, reference_list):
    return model.encode(reference_list, convert_to_tensor=True, show_progress_bar=True)


def predict_substances_batch(df, model, reference_list, output_csv="predictions.csv"):
    
    reference_embeddings = encode_reference_list(model, reference_list)
    input_texts = df["input_text"].tolist()
    input_embeddings = model.encode(input_texts, convert_to_tensor=True, show_progress_bar=True)
    
    predictions = []

    results = util.semantic_search(input_embeddings, reference_embeddings, top_k=1)

    for i, result in enumerate(results):
        best_match = reference_list[result[0]["corpus_id"]]
        similarity_score = round(result[0]["score"], 2)
        predictions.append({
            "input_text": input_texts[i],
            "label": df["label"].iloc[i],
            "predicted_label": best_match,
            "similarity" : similarity_score
        })

    predictions_df = pd.DataFrame(predictions) #to add: option to delete predictions under certain threshold
    predictions_df.to_csv(output_csv, index=False, sep=";")
    print(f"Predictions saved to {output_csv}")

### apply to data
df = load_data("subs_labels.csv")
reference_list = load_reference_list("substanz_referenz.csv")
train_df, val_df = train_test_split(df, test_size=0.2)
train_examples = prepare_train_examples(train_df, add_negative_samples=False)
val_examples = prepare_train_examples(val_df, add_negative_samples=True)
 
   
model = train_model(train_examples, val_examples, epochs=3)


predict_substances_batch(df, model, reference_list)


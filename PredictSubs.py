import pandas as pd
import numpy as np
import random
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from typing import List, Tuple

# define functions to read in data and reference list with valid substances
def load_data(csv_path: str) -> pd.DataFrame:
    """csv file with semicolon as separator, encoded in utf8, text should be in a column named input_text"""
    return pd.read_csv(csv_path, encoding="utf-8", sep=";")

def load_reference_list(reference_csv_path: str) -> List[str]:
    """csv file with semicolon as separator, encoded in utf8, requires column named Substanz"""
    df = pd.read_csv(reference_csv_path, encoding="utf-8", sep=";")
    return df["Substanz"].str.strip().unique().tolist()

def prepare_train_examples(df: pd.DataFrame, add_negative_samples: bool = False) -> List[InputExample]:
    """Labeled training examples for model

    Args:
        df: DataFrame with columns named 'input_text' and 'label'
        add_negative_samples: For evaluation, wrong pairs with label 0

    Returns:
        A list of examples for model training
    """
    train_examples = [InputExample(texts=[row["input_text"], row["label"]], label=1.0) for _, row in df.iterrows()]
    
    if add_negative_samples:
        unique_labels = df["label"].unique().tolist()
        for _, row in df.iterrows():
            negative_label = random.choice([lbl for lbl in unique_labels if lbl != row["label"]])
            train_examples.append(InputExample(texts=[row["input_text"], negative_label], label=0.0)) 
    
    return train_examples

# Training of model

def train_model(
    train_examples: List[InputExample], 
    val_examples: List[InputExample], 
    model_name: str = "multi-qa-mpnet-base-cos-v1", #todo: try clinicalBioBERT
    epochs: int = 3, 
    batch_size: int = 16
) -> SentenceTransformer:
    """Train model for substance extraction

    Args:
        train_examples: Training dataset as InputExample list
        val_examples: Validation dataset as InputExample list
        model_name: Name of the model by huggingface
        epochs: Number of training epochs
        batch_size: Batch size for training

    Returns:
        Trained model
    """
    model = SentenceTransformer(model_name)
    train_loss = losses.CosineSimilarityLoss(model)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(val_examples, name="val-eval")
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        evaluation_steps=100,
        output_path="substance_extractor_model",
    )
    return model

# predict based on trained model
def encode_reference_list(model: SentenceTransformer, reference_list: List[str]):
    """encode valid substances from list into embeddings"""
    return model.encode(reference_list, convert_to_tensor=True, show_progress_bar=True)

def predict_substance(
    input_text: str, 
    model: SentenceTransformer, 
    reference_list: List[str]
) -> Tuple[str, float]:
    """predict the most similar substance from the ref list

    Args:
        input_text: text reported to cancer registry
        model: trained model
        reference_list: List of valid substances

    Returns:
        best match and similarity score
    """
    reference_embeddings = encode_reference_list(model, reference_list)
    query_embedding = model.encode(input_text, convert_to_tensor=True)

    results = util.semantic_search(query_embedding, reference_embeddings, top_k=1)
    best_match = reference_list[results[0][0]["corpus_id"]]
    similarity_score = round(results[0][0]["score"], 2)

    return best_match, similarity_score

def predict_substances_batch(
    df: pd.DataFrame, 
    model: SentenceTransformer, 
    reference_list: List[str], 
    output_csv: str = "predictions.csv", 
    threshold: float = 0.8
) -> None:
    """predict substance

    Args:
        df: DataFrame containing input_text
        model: Trained model
        reference_list: ref list with substances
        output_csv: csv file path
        threshold: minimum similarity score for a prediction
    """
    
    reference_embeddings = encode_reference_list(model, reference_list)

    input_texts = df["input_text"].tolist()
    input_embeddings = model.encode(input_texts, convert_to_tensor=True, show_progress_bar=True)

    predictions = []
    #todo: try FAISS instead of semantic_search and compare results and speed
    results = util.semantic_search(input_embeddings, reference_embeddings, top_k=1)

    for i, result in enumerate(results):
        best_match = reference_list[result[0]["corpus_id"]]
        similarity_score = round(result[0]["score"], 2)
        
        # threshold defines what will be included in output
        if similarity_score < threshold:
            best_match = None

        predictions.append({
            "input_text": input_texts[i],
            "label": df["label"].iloc[i],
            "predicted_label": best_match,
            "similarity": similarity_score
        })

    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(output_csv, index=False, sep=";")
    print(f"Predictions saved to {output_csv}")

# run it

if __name__ == "__main__":
    df = load_data("subs_labels.csv")
    reference_list = load_reference_list("substanz_referenz.csv")
    train_df, val_df = train_test_split(df, test_size=0.2)
    train_examples = prepare_train_examples(train_df, add_negative_samples=False)
    val_examples = prepare_train_examples(val_df, add_negative_samples=True)
    model = train_model(train_examples, val_examples, epochs=3)
    predict_substances_batch(df, model, reference_list, output_csv="predicted_substances.csv", threshold=0.8)

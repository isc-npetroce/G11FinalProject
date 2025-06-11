import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import ast

# loads our final project dataset from HuggingFace
def load_dataset_from_hf():
    # Grab dataset CSV from Aditi's repository
    dataset_url = "https://huggingface.co/datasets/asarathy/patient_encounters1_notes/raw/main/patient_encounters1_notes.csv"
    df = pd.read_csv(dataset_url, on_bad_lines="warn")

    df['CLINICAL_NOTES_NONULL'] = df['CLINICAL_NOTES'].fillna('')
    df['CLINICAL_NOTES_LEMMATIZED_JOINED'] = df['CLINICAL_NOTES_CLEAN_LEMMATIZED'].apply(lambda x: ' '.join(ast.literal_eval(x)) if isinstance(x, str) else '')
    return df

# Loads data from the given dataframe into the vector DB, as in workshop 7.
def load_data_into_vectordb(df, data_column = "CLINICAL_NOTES_NONULL"):
    # starting with all-MiniLM as this was used in week 7
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.Client(Settings(
        persist_directory="./chroma_db"
    ))
    
    collection = client.get_or_create_collection(name=data_column)

    texts = df[data_column].astype(str).tolist()
    embeddings = model.encode(texts).tolist()

    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=[str(i) for i in range(len(texts))]
    )

    
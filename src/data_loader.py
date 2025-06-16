import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import ast

# helper function to wrap both load_dataset_from_hf and load_data_into_vectordb 
def load_vector_db(data_column = "CLINICAL_NOTES_NONULL", metadata_columns = ['PATIENT_ID', 'FIRST', 'ENCOUNTER_ID']):
    dataset = load_dataset_from_hf()
    collection = load_data_into_vectordb(dataset, data_column, metadata_columns)
    return collection

# loads our final project dataset from HuggingFace
def load_dataset_from_hf():
    # Grab dataset CSV from Aditi's repository
    dataset_url = "https://huggingface.co/datasets/asarathy/patient_encounters1_notes/raw/main/patient_encounters1_notes.csv"
    df = pd.read_csv(dataset_url, on_bad_lines="warn")

    # get two potential data columns
    # first is just the basic clinical notes with nulls replaced by emptystring
    df['CLINICAL_NOTES_NONULL'] = df['CLINICAL_NOTES'].fillna('')
    # second is our cleaned notes from Week 6. Unfortunately our tokenization breaks the input format for the sentence transformer,
    # so we naively join the cleaned text back into single documents by joining on ' '.
    df['CLINICAL_NOTES_LEMMATIZED_JOINED'] = df['CLINICAL_NOTES_CLEAN_LEMMATIZED'].apply(lambda x: ' '.join(ast.literal_eval(x)) if isinstance(x, str) else '')
    return df

# Loads data from the given dataframe into the vector DB, as in workshop 7.
# https://docs.trychroma.com/docs/collections/delete-data
def load_data_into_vectordb(df, data_column = "CLINICAL_NOTES_NONULL", metadata_columns = ['PATIENT_ID', 'FIRST', 'ENCOUNTER_ID']):
    # starting with all-MiniLM as this was used in week 7
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.Client(Settings(
        persist_directory="./chroma_db"
    ))
    
    collection = client.get_or_create_collection(name=data_column)

    texts = df[data_column].astype(str).tolist()
    # gets dict of metadata fields in 'records' format, which is what chroma expects
    # i.e. [{'index': 0, 'patient_id': 1, 'first': 'foo'}, {'index':1, 'patient_id':2, 'first': bar}]
    metadatas = df[metadata_columns].to_dict('records')

    embeddings = model.encode(texts).tolist()

    # use 'upsert' instead of add
    # this updates existing documents by ID and inserts if they do not exist
    collection.upsert(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=[str(i) for i in range(len(texts))]
    )
    # for convenience return the collection for querying
    return collection


# utility to get connection to chromaDB
# returns both client and collection object
# will throw exception if load_data_into_vectordb has not been called at least once
def get_chroma_client_and_collection(data_column = "CLINICAL_NOTES_NONULL"):
    client = chromadb.Client(Settings(
        persist_directory="./chroma_db"
    ))
    collection = client.get_collection(name=data_column)
    return client, collection

    
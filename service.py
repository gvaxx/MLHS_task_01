
from fastapi import FastAPI, File, UploadFile, HTTPException

from pydantic import BaseModel
from typing import List
import pandas as pd
import re
import pickle
import sklearn
import logging
from typing import Annotated
import io
from fastapi.responses import StreamingResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

regexp = r'\d+\.?\,?\d*'
newton_to_kg = 9.80665

model = None
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

encoder = None
with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

scaler = None
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str 
    engine: str
    max_power: str
    torque: str
    seats: float

class Items(BaseModel):
    objects: List[Item]

def string_to_float(s):
    return float(s.replace(',', '.'))

def extract_float(df_string):
    if pd.isna(df_string):
        return df_string
    findings = re.findall(regexp, df_string)    
    return string_to_float(findings[0]) if len(findings) != 0  else None

def extract_torque(string):
    if pd.isna(string):
        return string, string
    findings = re.findall(regexp, string)
    if re.search(r'kg|Kg|KG', string):
        return string_to_float(findings[0]) * newton_to_kg, string_to_float(findings[-1])
    return string_to_float(findings[0]), string_to_float(findings[-1])

def cast_torque_and_max_torque_rpm(df):
    new_columns = pd.DataFrame(df['torque'].apply(extract_torque).tolist(), columns=['torque', 'max_torque_rpm'])
    df['torque'] = new_columns['torque']
    df['max_torque_rpm'] = new_columns['max_torque_rpm']

def encode_onehot(df, columns):
    encoded = encoder.transform(df[columns])
    encoded_df =pd.DataFrame(encoded, columns=encoder.get_feature_names_out(columns))
    df = df.drop(columns, axis=1)
    df = pd.concat([encoded_df, df], axis=1)
    return df

def cast_columns(df):
    for column in ['mileage', 'engine', 'max_power']:
        df[column] = df[column].apply(extract_float)
    cast_torque_and_max_torque_rpm(df)
    return df

def data_preprocessing(df):
    df = df.drop(['selling_price', 'name'], axis=1)
    df = cast_columns(df)
    df = encode_onehot(df, ['fuel', 'seller_type', 'transmission', 'owner', 'seats'])
    df = scaler.transform(df)
    return df

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    df = pd.DataFrame([item.dict()])
    df = data_preprocessing(df)
    prediction = model.predict(df)
    return prediction[0]


@app.post("/predict_items")
def predict_items(file: UploadFile = File(...)):
    original_df = pd.read_csv(file.file)
    df = original_df.copy()
    for index, row in df.iterrows():
        try:
            Item(**row.to_dict())
        except Exception as e:
            raise HTTPException(detail=f'Error in row {index}: {e}', status_code=403)
    df = data_preprocessing(df)
    predictions = model.predict(df)
    predictions_df = pd.DataFrame(predictions, columns=['predicted_selling_price'])
    new_df = pd.concat([original_df, predictions_df], axis=1)
    stream = io.StringIO()
    new_df.to_csv(stream, index=True)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=predicted_price.csv"
    return response


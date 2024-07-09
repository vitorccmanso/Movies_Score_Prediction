import pandas as pd
from fastapi import FastAPI, Form, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.pipeline import PredictPipeline, CustomData

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")
pipeline = PredictPipeline()

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/manual_predict", response_class=HTMLResponse)
def render_manual_form(request: Request):
    return templates.TemplateResponse("manual_predict.html", {"request": request})

@app.post("/manual_predict", response_class=HTMLResponse)
def manual_predict(request: Request, released_year: int = Form(...),
                    certificate: str = Form(...),
                    runtime: int = Form(...),
                    genre: str = Form(...),
                    meta_score: float = Form(...),
                    director: str = Form(...),
                    star1: str = Form(...),
                    star2: str = Form(...),
                    star3: str = Form(...),
                    star4: str = Form(...),
                    no_of_votes: int = Form(...),
                    gross: int = Form(...)):
    custom_data = CustomData(released_year, certificate, runtime, genre, meta_score, director, star1, star2, star3, star4, no_of_votes, gross)
    data_df = custom_data.get_data_as_dataframe()
    prediction = pipeline.predict(data_df, manual=True)
    return templates.TemplateResponse("manual_predict.html", {"request": request, "predicted_rating": prediction})

@app.get("/dataset_predict", response_class=HTMLResponse)
def render_dataset_form(request: Request):
    return templates.TemplateResponse("dataset_predict.html", {"request": request})

@app.post("/dataset_predict", response_class=HTMLResponse)
def predict_dataset(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        processed_df = pipeline.process_dataset(df)
        predictions = pipeline.predict(processed_df)
        return templates.TemplateResponse("dataset_predict.html", {"request": request, "predicted_ratings": predictions})
    except ValueError as e:
        return templates.TemplateResponse("dataset_predict.html", {"request": request, "error_message": str(e)})
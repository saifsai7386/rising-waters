# Rising Waters – Flood Risk Prediction App

A Flask web application that predicts flood risk from 12 monthly rainfall measurements using a pre-trained machine learning model.

## Project structure

```
.
+-- app.py                      # Flask application
+-- data/
�   +-- generate_dataset.py     # builds the rainfall/flood dataset
�   +-- rainfall_data.csv       # generated dataset with monthly rainfall values
+-- models/
�   +-- floods.save             # pickled model bundle used by the app
+-- notebooks/
�   +-- eda.py                  # exploratory analysis and plots generation
�   +-- plots/                  # generated charts and statistics output
+-- preprocessing.py            # data cleaning, scaling, and preprocessing utilities
+-- train_model.py              # trains classifiers and selects the best model
+-- requirements.txt            # Python dependencies
+-- static/
�   +-- style.css               # application styling
+-- templates/
    +-- home.html               # landing page
    +-- predict.html            # rainfall input form
    +-- flood_result.html       # flood-risk result page
    +-- no_flood_result.html    # no-flood result page
```

## Environment setup

Use a Python 3.10+ environment:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the app

```bash
python app.py
```

Open `http://127.0.0.1:5000/` in your browser to use the application.

## Notes

- The deployed model bundle is stored in `models/floods.save`.
- The prediction form now accepts 12 monthly rainfall values plus a cloud visibility value in kilometres.
- Use `train_model.py` to retrain or compare classifiers if you update the dataset.
- The app is designed for local development and demonstration.

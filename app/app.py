from flask import Flask, request, render_template, send_from_directory, jsonify
import os
import tempfile
import pickle
import pandas as pd
from google.cloud import storage

app = Flask(__name__, static_folder='static', template_folder='templates')


def download_model_from_gcs(gcs_path: str, project_id: str) -> str:
    # gcs_path expected like gs://<bucket>/<path>/model.pkl
    if not gcs_path.startswith('gs://'):
        raise ValueError('gcs_path must start with gs://')

    parts = gcs_path[5:].split('/', 1)
    bucket_name = parts[0]
    blob_path = parts[1] if len(parts) > 1 else ''

    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    fd, local_path = tempfile.mkstemp(suffix='.pkl')
    os.close(fd)
    blob.download_to_filename(local_path)
    return local_path


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Expect a file upload (CSV) and query param project_id & model_gcs
    # If model_gcs not provided in form, use MODEL_GCS env var
    project_id = request.form.get('project_id')
    model_gcs = request.form.get('model_gcs') or os.environ.get('MODEL_GCS')
    
    if not model_gcs:
        return jsonify({'error': 'No model_gcs provided and MODEL_GCS env var not set'}), 400
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save uploaded file temporarily
    df = pd.read_csv(f)
    
    # Remove 'Class' column if present (it's the target, not a feature)
    if 'Class' in df.columns:
        df = df.drop('Class', axis=1)

    # Download model
    model_path = download_model_from_gcs(model_gcs, project_id)
    with open(model_path, 'rb') as fh:
        model = pickle.load(fh)

    # Predict (probability > 0.5)
    preds = model.predict(df)
    results = []
    for i, row in df.iterrows():
        r = row.to_dict()
        r['prediction'] = int(preds[i])
        results.append(r)

    return jsonify({'results': results})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)

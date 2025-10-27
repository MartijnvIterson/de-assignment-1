"""
Visual Kubeflow Pipeline for Vertex AI
Creates the visual pipeline representation like shown in the screenshot
"""
from kfp import dsl
from kfp.dsl import component, pipeline, Input, Output, Dataset, Model, Metrics
from kfp.dsl import Condition

@component(
    base_image="uvtmartijn/fraud-detection-components:latest"
)
def load_creditcard_data(
    project_id: str,
    dataset_output: Output[Dataset]
):
    """Load creditcard dataset from GCS"""
    import pandas as pd
    from google.cloud import storage
    
    print('Loading creditcard dataset from GCS...')
    
    # Load dataset from GCS
    storage_client = storage.Client()
    bucket = storage_client.bucket(f'{project_id}-data')
    blob = bucket.blob('creditcard.csv')
    
    # Download and load data
    blob.download_to_filename('/tmp/creditcard.csv')
    df = pd.read_csv('/tmp/creditcard.csv')
    
    # Save to pipeline artifact
    df.to_csv(dataset_output.path, index=False)
    
    print(f'Dataset loaded: {len(df)} rows, {df["Class"].sum()} fraud cases')

@component(
    base_image="uvtmartijn/fraud-detection-components:latest"
)
def preprocess_data(
    dataset_input: Input[Dataset],
    train_data: Output[Dataset],
    test_data: Output[Dataset]
):
    """Preprocess and split the data"""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    print('Preprocessing data...')
    
    # Load data
    df = pd.read_csv(dataset_input.path)
    
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save train data
    train_df = X_train.copy()
    train_df['Class'] = y_train
    train_df.to_csv(train_data.path, index=False)
    
    # Save test data
    test_df = X_test.copy()
    test_df['Class'] = y_test
    test_df.to_csv(test_data.path, index=False)
    
    print(f'Data split: {len(X_train)} train, {len(X_test)} test samples')



@component(
    base_image="uvtmartijn/fraud-detection-components:latest"
)
def train_xgboost_model(
    train_data: Input[Dataset],
    model_output: Output[Model]
):
    """Train XGBoost fraud detection model"""
    import pandas as pd
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    import pickle
    
    print('Training XGBoost model...')
    
    # Load training data
    train_df = pd.read_csv(train_data.path)
    X_train = train_df.drop('Class', axis=1)
    y_train = train_df['Class']
    
    # Train model
    model = xgb.XGBClassifier(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        scale_pos_weight=1
    )
    
    model.fit(X_train, y_train)
    
    # Calculate training metrics
    y_pred = model.predict(X_train)
    accuracy = accuracy_score(y_train, y_pred)
    precision = precision_score(y_train, y_pred, zero_division=0)
    recall = recall_score(y_train, y_pred, zero_division=0)
    
    # Save model
    with open(model_output.path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f'Model trained - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')

@component(
    base_image="uvtmartijn/fraud-detection-components:latest"
)
def evaluate_model(
    model_input: Input[Model],
    test_data: Input[Dataset],
    metrics_output: Output[Metrics]
) -> float:
    """Evaluate model performance"""
    import pandas as pd
    import pickle
    from sklearn.metrics import roc_auc_score
    import json
    
    print('Evaluating model...')
    
    # Load model and test data
    with open(model_input.path, 'rb') as f:
        model = pickle.load(f)
    
    test_df = pd.read_csv(test_data.path)
    X_test = test_df.drop('Class', axis=1)
    y_test = test_df['Class']
    
    # Make predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Save metrics (no approval decision here, that happens in comparison)
    metrics = {
        'roc_auc': roc_auc,
        'evaluation_timestamp': str(pd.Timestamp.now())
    }
    
    with open(metrics_output.path, 'w') as f:
        json.dump(metrics, f)
    
    print(f'ROC-AUC: {roc_auc:.4f} - Metrics saved for comparison')
    
    return roc_auc

@component(
    base_image="uvtmartijn/fraud-detection-components:latest"  
)
def check_model_improvement(
    current_roc_auc: float,
    project_id: str
) -> str:
    """Check if current model is better than deployed model. Returns 'true' or 'false'"""
    from google.cloud import aiplatform
    
    print(f'Current model ROC-AUC: {current_roc_auc:.4f}')
    
    # Initialize Vertex AI
    region = 'us-central1'
    aiplatform.init(project=project_id, location=region)
    
    try:
        # List existing models
        models = aiplatform.Model.list(
            filter='display_name="fraud-detection-model"',
            order_by='create_time desc'
        )
        
        if not models:
            print('No existing model found, this will be the first deployment')
            return 'true'
        
        # Get latest model
        latest_model = models[0]
        deployed_roc_auc = None
        
        # Try labels first (safer and consistent)
        try:
            labels = latest_model.labels or {}
        except Exception:
            labels = {}

        if labels and 'roc_auc' in labels:
            try:
                # Label format is with underscore: 0_9876 -> 0.9876
                roc_auc_str = labels['roc_auc'].replace('_', '.')
                deployed_roc_auc = float(roc_auc_str)
                print(f'Currently deployed model ROC-AUC (from labels): {deployed_roc_auc:.4f}')
            except Exception:
                deployed_roc_auc = None

        # Fallback: try to extract ROC-AUC from the description text
        if deployed_roc_auc is None:
            description = latest_model.description or ""
            import re
            auc_match = re.search(r'ROC AUC: ([\d\.]+)', description)
            if auc_match:
                try:
                    deployed_roc_auc = float(auc_match.group(1))
                    print(f'Currently deployed model ROC-AUC (from description): {deployed_roc_auc:.4f}')
                except Exception:
                    deployed_roc_auc = None

        if deployed_roc_auc is not None:
            if current_roc_auc <= deployed_roc_auc:
                print(f'New model ({current_roc_auc:.4f}) NOT better than deployed ({deployed_roc_auc:.4f})')
                return 'false'
            else:
                print(f'New model ({current_roc_auc:.4f}) IS better than deployed ({deployed_roc_auc:.4f})!')
                return 'true'
        else:
            print('Could not determine deployed model performance, proceeding with registration')
            return 'true'
        
    except Exception as e:
        print(f'Check failed: {e}, defaulting to registration')
        return 'true'


@component(
    base_image="uvtmartijn/fraud-detection-components:latest"  
)
def register_model(
    model_input: Input[Model],
    current_roc_auc: float,
    project_id: str,
    model_version: str
) -> str:
    """Register model in Vertex AI Model Registry and return GCS path"""
    from google.cloud import aiplatform, storage
    import json
    import pandas as pd
    
    print(f'Registering model with ROC-AUC: {current_roc_auc:.4f}')
    
    # Initialize Vertex AI
    region = 'us-central1'
    aiplatform.init(project=project_id, location=region)
    
    try:
        # Copy model to GCS location in Vertex AI compatible format
        storage_client = storage.Client()
        bucket = storage_client.bucket(f'{project_id}-models')
        
        # Create model directory structure for Vertex AI
        model_dir = f'fraud-model-{model_version}'
        
        # Upload model file
        model_blob = bucket.blob(f'{model_dir}/model.pkl')
        model_blob.upload_from_filename(model_input.path)
        
        # Create model metadata for Vertex AI (optional but helpful)
        model_metadata = {
            'framework': 'xgboost',
            'model_type': 'classification',
            'roc_auc': current_roc_auc,
            'version': model_version,
            'created_at': str(pd.Timestamp.now())
        }
        
        metadata_blob = bucket.blob(f'{model_dir}/metadata.json')
        metadata_blob.upload_from_string(json.dumps(model_metadata))
        
        # Build the GCS path to the model file
        model_gcs_path = f'gs://{project_id}-models/{model_dir}/model.pkl'
        print(f'Model uploaded to {model_gcs_path}')
        
        # Register model in Vertex AI Model Registry
        # Format ROC-AUC for label (no dots allowed - use underscore: 0.9876 -> 0_9876)
        roc_auc_label = f'{current_roc_auc:.4f}'.replace('.', '_')
        
        model_upload = aiplatform.Model.upload(
            display_name=f'fraud-detection-model',
            artifact_uri=f'gs://{project_id}-models/fraud-model-{model_version}/',
            serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest',
            description=f'Credit card fraud detection model - ROC AUC: {current_roc_auc:.4f} - Version: {model_version}',
            labels={
                'roc_auc': roc_auc_label,
                'version': str(model_version).replace('.', '-')
            },
            version_aliases=[f'v{model_version}', 'champion'],
            version_description=f'XGBoost fraud detection model with ROC-AUC: {current_roc_auc:.4f}'
        )
        
        print(f'Model registered in Vertex AI Model Registry: {model_upload.resource_name}')
        
        # Return the GCS path so it can be used by UI
        return model_gcs_path
        
    except Exception as e:
        print(f'Registration failed: {e}')
        return ''


@component(
    base_image="uvtmartijn/fraud-detection-components:latest"
)
def deploy_and_schedule_teardown(
    project_id: str,
    image_uri: str,
    service_name: str,
    model_gcs: str,
    region: str = 'us-central1',
    teardown_function_url: str = '',
    tasks_queue: str = '',
    tasks_location: str = 'us-central1',
    delay_seconds: int = 1800
) -> str:
    """Deploy image to Cloud Run and schedule a Cloud Tasks HTTP call to teardown function."""
    import subprocess
    import json

    # Deploy service with MODEL_GCS environment variable
    try:
        subprocess.check_call([
            'gcloud', 'run', 'deploy', service_name,
            '--image', image_uri,
            '--platform', 'managed',
            '--region', region,
            '--allow-unauthenticated',
            '--set-env-vars', f'MODEL_GCS={model_gcs}',
            '--project', project_id,
            '--quiet'
        ])
    except Exception as e:
        return f'deploy failed: {e}'

    # Get service URL
    try:
        url = subprocess.check_output([
            'gcloud', 'run', 'services', 'describe', service_name,
            '--platform', 'managed', '--region', region,
            '--format', 'value(status.url)', '--project', project_id
        ]).decode().strip()
    except Exception:
        url = ''

    # If teardown scheduling info provided, create Cloud Tasks task
    if teardown_function_url and tasks_queue:
        try:
            # Import only when needed (so it doesn't fail if google-cloud-tasks not installed)
            from google.cloud import tasks_v2
            from google.protobuf import timestamp_pb2
            import datetime

            client = tasks_v2.CloudTasksClient()
            parent = client.queue_path(project_id, tasks_location, tasks_queue)

            d = datetime.datetime.utcnow() + datetime.timedelta(seconds=delay_seconds)
            timestamp = timestamp_pb2.Timestamp()
            timestamp.FromDatetime(d)

            payload = json.dumps({'project_id': project_id, 'service_name': service_name, 'region': region}).encode()
            task = {
                'http_request': {
                    'http_method': tasks_v2.HttpMethod.POST,
                    'url': teardown_function_url,
                    'headers': {'Content-Type': 'application/json'},
                    'body': payload
                },
                'schedule_time': timestamp
            }
            client.create_task(parent=parent, task=task)
        except Exception as e:
            return f'deploy_ok_but_schedule_failed: {e}'

    return f'deployed: {url}'

@pipeline(
    name="fraud-detection-pipeline",
    description="Credit Card Fraud Detection MLOps Pipeline"
)
def fraud_detection_pipeline(
    project_id: str = "data-engineering-jads-2025",
    model_version: str = "latest",
    dockerhub_username: str = "uvtmartijn",
    teardown_function_url: str = '',
    tasks_queue: str = '',
    tasks_location: str = 'us-central1'
):
    """
    Visual fraud detection pipeline for Vertex AI
    This creates the visual representation shown in the screenshot
    """
    
    # Step 1: Load data
    load_data_task = load_creditcard_data(project_id=project_id)
    load_data_task.set_display_name("Load Dataset")
    
    # Step 2: Preprocess data
    preprocess_task = preprocess_data(
        dataset_input=load_data_task.outputs["dataset_output"]
    )
    preprocess_task.set_display_name("Preprocess Data")
    preprocess_task.after(load_data_task)
    
    # Step 3: Train model
    train_task = train_xgboost_model(
        train_data=preprocess_task.outputs["train_data"]
    )
    train_task.set_display_name("Train XGBoost Model")
    train_task.after(preprocess_task)
    
    # Step 4: Evaluate model
    evaluate_task = evaluate_model(
        model_input=train_task.outputs["model_output"],
        test_data=preprocess_task.outputs["test_data"]
    )
    evaluate_task.set_display_name("Evaluate Model")
    evaluate_task.after(train_task)
    
    # Step 5: Check if model is improvement
    check_task = check_model_improvement(
        current_roc_auc=evaluate_task.outputs['Output'],  # Use named output explicitly
        project_id=project_id
    )
    check_task.set_display_name("Check Model Improvement")
    check_task.after(evaluate_task)
    
    # Step 6: Conditionally register model (only if better)
    with Condition(check_task.output == 'true'):
        register_task = register_model(
            model_input=train_task.outputs["model_output"],
            current_roc_auc=evaluate_task.outputs['Output'],  # Use named output explicitly
            project_id=project_id,
            model_version=model_version
        )
        register_task.set_display_name("Register Model")
        register_task.after(check_task)
    
    # Step 7: Always deploy UI (not dependent on registration)
    # Use the newly trained model path
    service_name = f'fraud-ui-{model_version}'
    image_uri = f'{dockerhub_username}/fraud-ui:{model_version}'
    model_gcs_path = f'gs://{project_id}-models/fraud-model-{model_version}/model.pkl'
    
    deploy_task = deploy_and_schedule_teardown(
        project_id=project_id,
        image_uri=image_uri,
        service_name=service_name,
        model_gcs=model_gcs_path,
        region='us-central1',
        teardown_function_url=teardown_function_url,
        tasks_queue=tasks_queue,
        tasks_location=tasks_location,
        delay_seconds=1800
    )
    deploy_task.set_display_name('Deploy UI to Cloud Run')
    deploy_task.after(train_task)  # Deploy after training (not after registration)

if __name__ == "__main__":
    # This can be used for local compilation testing
    from kfp import compiler
    
    compiler.Compiler().compile(
        pipeline_func=fraud_detection_pipeline,
        package_path="fraud_detection_pipeline.json"
    )
    print("Pipeline compiled successfully!")
    print("Generated: fraud_detection_pipeline.json")
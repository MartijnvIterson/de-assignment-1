"""
Compile Visual Pipeline for Vertex AI
Compiles the KFP pipeline to JSON for Vertex AI deployment
"""
import os
import sys

def compile_pipeline():
    """Compile the visual pipeline"""
    try:
        import sys
        import os
        
        from kfp import compiler
        from visual_pipeline import fraud_detection_pipeline
        
        print("Compiling visual pipeline...")
        
        # Compile pipeline
        compiler.Compiler().compile(
            pipeline_func=fraud_detection_pipeline,
            package_path="fraud_detection_pipeline.json"
        )
        
        # Check if file was created
        if os.path.exists("fraud_detection_pipeline.json"):
            file_size = os.path.getsize("fraud_detection_pipeline.json")
            print(f"Pipeline compiled successfully!")
            print(f"Generated: fraud_detection_pipeline.json ({file_size} bytes)")
            return True
        else:
            print("Pipeline compilation failed - no output file")
            return False
            
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure KFP is installed: pip install kfp==2.4.0")
        return False
    except Exception as e:
        print(f"Compilation error: {e}")
        return False

def run_pipeline_in_vertex_ai():
    """Submit the pipeline to Vertex AI"""
    try:
        from google.cloud import aiplatform
        
        project_id = os.getenv('PROJECT_ID', 'data-engineering-jads-2025')
        region = os.getenv('REGION', 'us-central1')
        short_sha = os.getenv('SHORT_SHA', 'latest')
        service_account = os.getenv('SERVICE_ACCOUNT', f"github-actions@{project_id}.iam.gserviceaccount.com")
        teardown_url = os.getenv('TEARDOWN_FUNCTION_URL', '')
        tasks_queue = os.getenv('TASKS_QUEUE', '')
        
        print(f"Submitting pipeline to Vertex AI...")
        print(f"   Project: {project_id}")
        print(f"   Region: {region}")
        print(f"   Version: {short_sha}")
        print(f"   Service Account: {service_account}")
        print(f"   Teardown URL: {teardown_url}")
        print(f"   Tasks Queue: {tasks_queue}")
        
        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=region)
        
        # Create pipeline job
        job = aiplatform.PipelineJob(
            display_name=f"fraud-detection-pipeline-{short_sha}",
            template_path="fraud_detection_pipeline.json",
            parameter_values={
                "project_id": project_id,
                "model_version": short_sha,
                "dockerhub_username": "uvtmartijn",
                "teardown_function_url": teardown_url,
                "tasks_queue": tasks_queue,
                "tasks_location": region
            },
            enable_caching=False
        )
        
        # Submit pipeline with service account
        job.submit(service_account=service_account)
        
        print(f"Pipeline submitted successfully!")
        print(f"View pipeline: https://console.cloud.google.com/vertex-ai/pipelines?project={project_id}")
        print(f"Pipeline ID: {job.resource_name}")
        
        return True
        
    except Exception as e:
        print(f"Failed to submit pipeline: {e}")
        return False

def main():
    """Main function"""
    print("Visual Pipeline Compiler for Vertex AI")
    print("=" * 50)
    
    # Step 1: Compile pipeline
    if not compile_pipeline():
        sys.exit(1)
    
    # Step 2: Submit to Vertex AI
    if not run_pipeline_in_vertex_ai():
        sys.exit(1)
    
    print("Visual pipeline deployed successfully!")
    print("Check Vertex AI Console to see the visual pipeline flow!")

if __name__ == "__main__":
    main()
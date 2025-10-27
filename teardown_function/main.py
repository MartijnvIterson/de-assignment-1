"""
Cloud Function to delete a Cloud Run service
Triggered by Cloud Tasks after a delay
"""
import functions_framework
from google.cloud import run_v2
import json

@functions_framework.http
def teardown_cloud_run(request):
    """HTTP Cloud Function to delete a Cloud Run service."""
    try:
        # Parse request
        request_json = request.get_json(silent=True)
        
        if not request_json:
            return ('Request must be JSON', 400)
        
        project_id = request_json.get('project_id')
        service_name = request_json.get('service_name')
        region = request_json.get('region', 'us-central1')
        
        if not project_id or not service_name:
            return ('Missing project_id or service_name', 400)
        
        print(f'Deleting Cloud Run service: {service_name} in {region}')
        
        # Delete the service
        client = run_v2.ServicesClient()
        name = f'projects/{project_id}/locations/{region}/services/{service_name}'
        
        try:
            operation = client.delete_service(name=name)
            operation.result()  # Wait for completion
            print(f'Successfully deleted service: {service_name}')
            return (json.dumps({'status': 'deleted', 'service': service_name}), 200)
        except Exception as delete_error:
            # Check if service doesn't exist (already deleted)
            if '404' in str(delete_error) or 'not found' in str(delete_error).lower():
                print(f'Service {service_name} already deleted or does not exist')
                return (json.dumps({'status': 'already_deleted', 'service': service_name}), 200)
            else:
                # Re-raise other errors
                raise
        
    except Exception as e:
        error_msg = f'Error deleting service: {str(e)}'
        print(error_msg)
        return (json.dumps({'status': 'error', 'message': str(e)}), 500)

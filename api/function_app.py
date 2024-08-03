import azure.functions as func
import logging
import json

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="message")
def message(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    response = {
        "text": "Hello from the API"
    }
    return func.HttpResponse(
        json.dumps(response),
        mimetype="application/json"
    )

#    name = req.params.get('name')
#    if not name:
#        try:
#            req_body = req.get_json()
#        except ValueError:
#            pass
#        else:
#            name = req_body.get('name')
#
#    if name:
#        return func.HttpResponse(f"Hello, {name}. This HTTP triggered function executed successfully.")
#    else:
#        return func.HttpResponse(
#             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
#             status_code=200
#        )

@app.route(route="upload_file_old", auth_level=func.AuthLevel.ANONYMOUS)
def upload_file_old(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    name = req.params.get('name')
    if not name:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            name = req_body.get('name')

    if name:
        return func.HttpResponse(f"Hello, {name}. This HTTP triggered function executed successfully.")
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )


@app.route(route="upload_file", auth_level=func.AuthLevel.ANONYMOUS)
def upload_file(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    try:
        # Get the uploaded file
        uploaded_file = req.files.get('file')

        if uploaded_file:
            # Process the file as needed
            file_content = uploaded_file.read()
            
            # Stub for additional file processing
            process_file(file_content)

            logging.info('File uploaded successfully.')
            return func.HttpResponse("File uploaded successfully.", status_code=200)
        else:
            return func.HttpResponse("No file uploaded.", status_code=400)
    except Exception as e:
        logging.error(f"Error during file upload: {str(e)}")
        return func.HttpResponse("File upload failed.", status_code=500)

def process_file(file_content):
    # Stub for additional functionality to process the uploaded file
    logging.info('Processing file...')
    # Add your file processing logic here

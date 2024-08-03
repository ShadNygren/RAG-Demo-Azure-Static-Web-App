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

@app.route(route="upload_file", auth_level=func.AuthLevel.ANONYMOUS)
def upload_file(req: func.HttpRequest) -> func.HttpResponse:
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
import json
import http.client
import boto3
import os
from botocore.exceptions import ClientError


def lambda_handler(event, context):
    try:
        print("Event received:", event)
        # property_object = event.get('body', '{}')
        property_object = event
        if not property_object:
            return {
                "statusCode": 400,
                "body": json.dumps(
                    {"error": "Invalid input, a body must be provided"}
                ),
            }
        print("Body received: ", property_object)

        # Validate required fields in Property
        required_fields = [
            'habitaciones',
            'vivienda',
            'aseos',
            'metros',
            'CUDIS',
        ]

        missing_fields = [
            field for field in required_fields if field not in property_object
        ]

        if missing_fields:
            return {
                "statusCode": 400,
                "body": json.dumps(
                    {
                        "error": f"Missing required fields: {', '.join(missing_fields)}"
                    }
                ),
            }

        CUDISsufix = (
            "-P"
            if property_object["vivienda"] == "PISO"
            else "-C"
            if property_object["vivienda"] == "CASA"
            else ""
        )
        CUDIS = str(property_object["CUDIS"]) + CUDISsufix
        data_by_CUDIS = get_data_by_CUDIS(CUDIS)

        if (
            not isinstance(property_object['habitaciones'], int)
            or property_object['habitaciones'] < 0
        ):
            return {
                "statusCode": 400,
                "body": json.dumps(
                    {"error": "'habitaciones' must be a positive integer"}
                ),
            }
        if (
            not isinstance(property_object['aseos'], int)
            or property_object['aseos'] < 0
        ):
            return {
                "statusCode": 400,
                "body": json.dumps(
                    {"error": "'aseos' must be a positive integer"}
                ),
            }
        if (
            not isinstance(property_object['metros'], int)
            or property_object['metros'] < 0
        ):
            return {
                "statusCode": 400,
                "body": json.dumps(
                    {"error": "'metros' must be a positive integer"}
                ),
            }

        # Validate optional field 'comodidades'
        valid_comodidades = ["GARAJE", "TERRAZA", "PISCINA"]

        if 'comodidades' in property_object:
            if not isinstance(property_object['comodidades'], list) or not all(
                isinstance(c, str) and c in valid_comodidades
                for c in property_object['comodidades']
            ):
                return {
                    "statusCode": 400,
                    "body": json.dumps(
                        {
                            "error": f"'comodidades' must be a list of valid values: {', '.join(valid_comodidades)}"
                        }
                    ),
                }

        comodidades_counter = len(set(property_object['comodidades']))

        # Define the internal API endpoint
        internal_api_url = "https://internal-api.example.com/process-property"
        # Make an HTTP POST request to the internal API using urllib
        data = {
            "Habitaciones": property_object['habitaciones'],
            "Aseos": property_object['aseos'],
            "Metros": property_object['metros'],
            "CUDIS": property_object['CUDIS'],
            "Poblacion": int(data_by_CUDIS['Poblacion']),
            "Renta bruta media por persona": int(
                data_by_CUDIS['Renta_bruta_media']
            ),
            "Comodidades": comodidades_counter,
            "Capital": 1 if data_by_CUDIS['Capital'] else 0,
            "Precio_medio_mun_tipo": int(data_by_CUDIS['Precio_medio']),
        }

        path1 = "/predict_xgb"
        path2 = "/predict_ann"
        method = "POST"
        headers = {'Content-Type': 'application/json'}
        body = json.dumps(data).encode('utf-8')

        try:
            print(f"Sending request to internal API: {internal_api_url}")
            print(f"Data to be sent to internal API: {data}")
            response1 = call_internal_api(method, path1, headers, body)
            response2 = call_internal_api(method, path2, headers, body)
            print("Response from internal API:", response1, response2)
            response = {
                "message": "Property processed successfully",
                "predictions": [response1['body'], response2['body']],
            }
            return response

        except Exception as e:
            return {
                "statusCode": 502,
                "body": json.dumps(
                    {
                        "error": "Failed to connect to internal API",
                        "details": str(e),
                    }
                ),
            }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps(
                {"error": "An error occurred", "details": str(e)}
            ),
        }


def get_data_by_CUDIS(CUDIS):
    # Define the DynamoDB table name
    table_name = 'cudisTable'
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)
    print(f"Fetching record with CUDIS={CUDIS} from DynamoDB")
    try:
        # Recuperamos el registro con ese CUDIS
        response = table.get_item(Key={'CUDIS': CUDIS})
        if 'Item' in response:
            print("Record fetched:", response['Item'])
            return response['Item']
        else:
            print(f"No record found with CUDIS={CUDIS}")
            return None
    except ClientError as e:
        print(f"Error fetching record: {e.response['Error']['Message']}")
        return None


def call_internal_api(method, path, headers, body):
    print(f"Calling internal API: {path}")
    ec2_host = os.getenv("ec2Url")
    ec2_port = os.getenv("ec2Port")
    conn = http.client.HTTPConnection(ec2_host, ec2_port)

    try:
        conn.request(method, path, body, headers)
        response = conn.getresponse()
        response_body = response.read().decode()
        return {
            "statusCode": response.status,
            "body": json.loads(response_body),
        }
    except Exception as e:
        return {"statusCode": 500, "body": None}

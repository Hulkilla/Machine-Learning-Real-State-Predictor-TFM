import json
import boto3
from botocore.exceptions import ClientError


def lambda_handler(event, context):

    bucket_name = 'fillcudistable'
    table_name = 'cudisTable'
    s3 = boto3.client('s3')
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)

    try:
        # Obtenemos el archivo del evento
        for record in event['Records']:
            s3_object_key = record['s3']['object']['key']
            print(f"Processing file: {s3_object_key}")

            # Obtenemos el archivo JSON
            response = s3.get_object(Bucket=bucket_name, Key=s3_object_key)
            file_content = response['Body'].read().decode('utf-8')
            records = json.loads(file_content)

            # Insertamos cada registro en dynamoDB
            for record in records:
                try:
                    # Procesamos el CUDIS segun el tipo de vivienda para convertirlo en una clave unica
                    sufix = (
                        "-P"
                        if record.get("Vivienda") == "Piso"
                        else "-C"
                        if record.get("Vivienda") == "Casa"
                        else ""
                    )
                    record['CUDIS'] = str(record['CUDIS']) + sufix

                    print(
                        f"Inserting record with CUDIS={record['CUDIS']} into DynamoDB"
                    )

                    table.put_item(Item=record)
                except ClientError as e:
                    print(
                        f"Error inserting record with CUDIS={record['CUDIS']}: {e.response['Error']['Message']}"
                    )

        return {
            "statusCode": 200,
            "body": json.dumps({"message": "File processed successfully"}),
        }

    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps(
                {"error": "An error occurred", "details": str(e)}
            ),
        }

import json
import http.client
import os


def lambda_handler(event, context):
    # Dirección de la máquina EC2

    # Reemplaza con tu dirección EC2
    ec2_host = os.getenv("ec2Url")

    # Cambia si usas HTTPS (443) o un puerto diferente
    ec2_port = os.getenv("ec2Port")

    # Crear conexión HTTP
    conn = http.client.HTTPConnection(ec2_host, ec2_port)

    # Extraer datos del evento recibido
    path = "/health"
    method = "GET"
    headers = event.get("headers", {})
    body = None

    # Realizar la solicitud a la máquina EC2
    try:
        conn.request(method, path, body, headers)
        response = conn.getresponse()
        response_body = response.read().decode()
        return {
            "statusCode": response.status,
            "body": json.loads(response_body),
        }
    except Exception as e:
        return {
            "statusCode": 404,
            "body": "La maquina no se encuentra disponible o no existe "
            + str(e),
        }

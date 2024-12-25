from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pymysql
from flask_cors import CORS  # Importar CORS
from dotenv import load_dotenv
import os
from datetime import datetime
# pip install flask transformers torch flask-cors
# pip install pymysql 
# pip install python-dotenv

# Cargar variables del archivo .env
load_dotenv()
# Configurar la conexión a la base de datos
db_config = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_DATABASE'),
}
  
# Cargar el modelo y el tokenizer entrenados
model_path = "bert_clasificador_espanol"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# Crear la aplicación Flask
app = Flask(__name__)

# Habilitar CORS para todas las rutas y dominios
CORS(app)

# Etiquetas de las clases
CLASSES = ["Positivo", "Negativo", "Neutro", "Invalido"]
PUNTAJES = {"Positivo": 5, "Negativo": 1, "Neutro": 3, "Invalido": 0}

# Función para obtener una conexión a la base de datos 
def obtener_conexion():
    try:
        # Intentamos conectar utilizando PyMySQL
        conexion = pymysql.connect(**db_config)
        
        # Verificamos si la conexión fue exitosa
        print("Conexión exitosa a la base de datos")
        cursor = conexion.cursor()
        return conexion, cursor  # Devolvemos la conexión y el cursor
    except pymysql.MySQLError as e:
        print(f"Error al conectar a la base de datos: {e}")
        return None, None  # Indicamos que la conexión falló

MAX_COMMENT_LENGTH = 128 
# Función de validación para los comentarios y otros campos
def validar_entrada(comentarios):
    # Validar que la entrada es una lista de comentarios
    if not comentarios or not isinstance(comentarios, list):
        return {"error": "Debe enviar una lista de comentarios."}, 400

    # Validar los campos de cada comentario
    for item in comentarios:
        comentario = item.get("user_comment")
        if not comentario:
            return {"error": "El campo 'user_comment' es obligatorio."}, 400

        # Validación: Verificar que el comentario no esté vacío
        if not comentario.strip():
            return {"error": "El comentario no puede estar vacío."}, 400

        # Validación: Verificar que el comentario no exceda la longitud máxima
        if len(comentario) > MAX_COMMENT_LENGTH:
            return {"error": f"El comentario no puede exceder los {MAX_COMMENT_LENGTH} caracteres."}, 400

        # Validación de los otros campos
        product_id = item.get("product_id")
        if not product_id:
            return {"error": "El campo 'product_id' es obligatorio."}, 400

        user_id = item.get("user_id")
        if not user_id:
            return {"error": "El campo 'user_id' es obligatorio."}, 400
    return None, 200  # Retorna None si todo está bien

#ESTO ES EL ENDPOINT Ruta para clasificar un comentario
@app.route('/clasificar', methods=['POST'])
def clasificar():
 try:
        # Obtener los comentarios desde la solicitud
        data = request.get_json()
        print(data)

        # Validar datos requeridos
        product_id = data.get('product_id')
        user_id = data.get('user_id')
        user_comment = data.get('user_comment')

        if not product_id or not user_id or not user_comment:
            return jsonify({"error": "Faltan datos necesarios en la solicitud"}), 400
        
        # Preprocesar comentarios
        resultados = []

        # Generar la fecha actual
        date_comment = datetime.now().strftime('%Y-%m-%d')

        # Tokenizar el comentario
        inputs = tokenizer(
                user_comment,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
        )

        # Obtener la predicción
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_class = torch.argmax(outputs.logits, dim=-1).item()
        
        # Obtener el nombre de la clase y su puntaje
        clase = CLASSES[predicted_class]
        puntaje = PUNTAJES[clase]

        # Añadir el resultado a la lista de resultados
        resultados.append({
            "product_id": product_id,
            "user_comment": user_comment,
            "date_comment": date_comment,
            "classification": clase,
            "rating": puntaje
        })
        
        # Conectar a la base de datos
        conexion, cursor = obtener_conexion()

        if conexion:
            try:
                # Insertar en la tabla de comentarios
                cursor.execute(
                    "INSERT INTO product_comment (product_id, user_id, user_comment, date_comment) "
                    "VALUES (%s, %s, %s, %s)",
                    (product_id, user_id, user_comment, date_comment)
                )
                conexion.commit()  # Confirmar cambios
                
                # Obtener el idprodcomment generado automáticamente
                idprodcomment = cursor.lastrowid  # Esto obtiene el ID autogenerado para MySQL

                print(f"Comentario insertado en la tabla de comentarios. ID: {idprodcomment}, Producto ID: {product_id}, Usuario ID: {user_id}, Comentario: {user_comment}, Fecha: {date_comment}")

                # Insertar en la tabla de auditoría
                cursor.execute(
                    "INSERT INTO audit_product_comment (idprodcomment, product_id, user_id, user_comment, classification, rating, date_audit) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (idprodcomment, product_id, user_id, user_comment, clase, puntaje, date_comment)
                )
                conexion.commit()  # Confirmar cambios
                print(f"Comentario insertado en la tabla de auditoría. ID: {idprodcomment}, Producto ID: {product_id}, Usuario ID: {user_id}, Comentario: {user_comment}, Clasificación: {clase}, Puntaje: {puntaje}, Fecha de auditoría: {date_comment}")

            except Exception as e:
                print(f"Error al insertar en la base de datos: {e}")
                return jsonify({"error": "Error al insertar el comentario en la base de datos"}), 500
            finally:
                # Cerrar la conexión a la base de datos
                conexion.close()

        # Retornar los resultados con la clasificación y puntaje
        return jsonify(resultados), 200  # Retornar los resultados clasificados y almacenados
 except Exception as e:
        # Manejo general de errores
        print(f"Error en el procesamiento: {e}")
        return jsonify({"error": "Hubo un problema procesando la solicitud"}), 500

if __name__ == '__main__':
    app.run(port=5000)

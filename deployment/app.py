from flask import Flask, request, render_template, jsonify
import torch
import boto3
import io
from torchvision import transforms
from PIL import Image

app = Flask(__name__)

# Initialize a session using Amazon S3
s3 = boto3.client('s3')

# Specify the bucket name and model key
bucket_name = 'your-bucket-name'
model_key = 'path/to/your/model.pth'

# Download the model file from S3
response = s3.get_object(Bucket=bucket_name, Key=model_key)
model_file = io.BytesIO(response['Body'].read())

# Load the model
model = torch.load(model_file)
model.eval()

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'})

    file = request.files['file']
    img = Image.open(io.BytesIO(file.read()))
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = outputs.max(1)
        prediction = predicted.item()

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)

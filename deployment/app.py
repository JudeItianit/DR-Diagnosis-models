from flask import Flask, request, render_template, jsonify
import torch
from torchvision import transforms
from PIL import Image
import io
import timm

app = Flask(__name__)

# Load the model
model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=5)
model.load_state_dict(torch.load('model.pth'))
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

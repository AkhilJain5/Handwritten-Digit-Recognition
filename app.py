from flask import Flask, request, render_template, redirect, url_for, flash
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
import base64
from io import BytesIO

app = Flask(__name__)
app.secret_key = 'secret_key'

# Define the models 
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 32 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the models
mlp_model = MLP()
mlp_model.load_state_dict(torch.load('C:/Users/hp/OneDrive/Documents/GitHub/Handwritten-Digit-Recognition/MNIST Models/mlp_model.pth'))
mlp_model.eval()

cnn_model = SimpleCNN()
cnn_model.load_state_dict(torch.load('C:/Users/hp/OneDrive/Documents/GitHub/Handwritten-Digit-Recognition/MNIST Models/simple_cnn_model.pth'))
cnn_model.eval()

lenet_model = LeNet5()
lenet_model.load_state_dict(torch.load('C:/Users/hp/OneDrive/Documents/GitHub/Handwritten-Digit-Recognition/MNIST Models/lenet5_model.pth'))
lenet_model.eval()

# Define the transformation to be applied to the input image
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def load_and_preprocess_image(image_path):
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def evaluate_single_image(model, image_path):
    image = load_and_preprocess_image(image_path)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        predicted_label = predicted.item()
    return predicted_label

def preprocess_canvas_image(image_data):
    image = Image.open(BytesIO(base64.b64decode(image_data.split(',')[1])))
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        if file:
            model_choice = request.form.get('model_choice')

            # Create the 'static/uploads' directory if it doesn't exist
            upload_folder = os.path.join('static', 'uploads')
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)

            image_path = os.path.join(upload_folder, file.filename)
            file.save(image_path)

            if model_choice == 'mlp':
                predicted_label = evaluate_single_image(mlp_model, image_path)
            elif model_choice == 'cnn':
                predicted_label = evaluate_single_image(cnn_model, image_path)
            elif model_choice == 'lenet':
                predicted_label = evaluate_single_image(lenet_model, image_path)
            else:
                flash('Invalid model choice', 'error')
                return redirect(request.url)

            return render_template('result.html', prediction=predicted_label, image_path=image_path)

    return render_template('index.html')

@app.route('/predict_canvas', methods=['POST'])
def predict_canvas():
    model_choice = request.form.get('model_choice_canvas')
    canvas_data = request.form.get('canvasData')

    if not canvas_data:
        flash('No drawing data', 'error')
        return redirect(url_for('index'))

    image = preprocess_canvas_image(canvas_data)

    if model_choice == 'mlp':
        model = mlp_model
    elif model_choice == 'cnn':
        model = cnn_model
    elif model_choice == 'lenet':
        model = lenet_model
    else:
        flash('Invalid model choice', 'error')
        return redirect(url_for('index'))

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        predicted_label = predicted.item()

    return render_template('result.html', prediction=predicted_label, image_path=None)

if __name__ == '__main__':
    app.run(debug=True)

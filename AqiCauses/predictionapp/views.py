from django.shortcuts import render
import torch
import torch.nn as nn
import numpy as np
import os
import math
from django.conf import settings
# Define model architecture
class CasesPrediction(nn.Module):
    def __init__(self):
        super(CasesPrediction, self).__init__()
        self.fc1 = nn.Linear(9, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

# Load the trained model
model = CasesPrediction()

# Load only weights
import os
model_path = os.path.join(os.path.dirname(__file__), 'case_prediction.pth')
with open(model_path, 'rb') as f:
    model.load_state_dict(torch.load(f, map_location='cpu'))

# View function
def input_data(request):
    context = {}
    if request.method == 'POST':
        try:
            # Get values safely
            fields = ['AQI', 'pm10', 'pm25', 'no2', 'so2', 'o3', 'temperature', 'humidity', 'windspeed']
            values = [float(request.POST.get(field, 0)) for field in fields]

            input_tensor = torch.tensor([values], dtype=torch.float32)
            output = model(input_tensor)

            respiratory_cases = math.ceil(output[0][0].item())
            cardio_cases = math.ceil(output[0][1].item())

            return render(request, 'output.html', {
                'respiratory_cases': respiratory_cases,
                'cardio_cases': cardio_cases
            })

        except ValueError as e:
            context['error'] = f"Prediction error: {e}"

    return render(request, 'index.html')
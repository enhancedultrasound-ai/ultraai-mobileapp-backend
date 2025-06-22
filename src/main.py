import torch
from appwrite.client import Client
from appwrite.services.storage import Storage
import io
import torchvision.transforms as transforms
import torchvision.models as models

# Define Data Transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
])

def main(context):
    # Initialize Appwrite client
    client = Client()
    client.set_endpoint(context.get('APPWRITE_ENDPOINT'))
    client.set_project(context.get('APPWRITE_FUNCTION_PROJECT_ID'))
    client.set_key(context.get('APPWRITE_API_KEY'))
    
    # Retrieve model
    storage = Storage(client)
    model_file_id = context.get('MODEL_FILE_ID')
    model_bytes = storage.get_file_download(model_file_id)
    buffer = io.BytesIO(model_bytes)
    
    # Load model
    model = models.mobilenet_v3_small(pretrained=True)
    model.load_state_dict(torch.load(buffer, map_location=torch.device('cpu')))
    model.eval()  # Critical for inference[2]
    
    # Process input
    input_data = context.req.body  # For POST requests
    input_tensor = torch.tensor(input_data)
    input_tensor = Image.fromarray(input_tensor.squeeze(), mode='L')
    input_tensor = transform(input_tensor).unsqueeze(0)
    
    
    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
    
    return context.res.json({
        "predictions": predicted.item().tolist(),
        "model_version": "1.0"
    })
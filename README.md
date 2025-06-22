# UltraAI Mobile App Backend - Image Inference

This is an Appwrite Cloud Function for image inference using MobileNet V3.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set up environment variables in Appwrite:
   - `APPWRITE_ENDPOINT`: Your Appwrite endpoint
   - `APPWRITE_FUNCTION_PROJECT_ID`: Your project ID
   - `APPWRITE_API_KEY`: Your API key with storage permissions
   - `MODEL_FILE_ID`: The ID of your trained model file in Appwrite storage

## Deployment

1. Deploy to Appwrite using the CLI:

```bash
appwrite deploy function
```

2. Or use the `appwrite.json` configuration file for deployment settings.

## Usage

Send a POST request to your function endpoint with one of these formats:

### Base64 encoded image:

```json
{
  "image": "base64_encoded_image_data_here"
}
```

### Image array:

```json
[255, 128, 64, ...]
```

## Response Format

Success:

```json
{
  "predictions": 123,
  "model_version": "1.0",
  "status": "success"
}
```

Error:

```json
{
  "error": "Error message",
  "status": "error"
}
```

## Notes

- The function expects grayscale images that will be converted to RGB
- Images are resized to 299x299 pixels
- Uses ImageNet normalization
- Model runs on CPU for compatibility

from django.shortcuts import render
from django.http import HttpResponse
from io import BytesIO
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import numpy as np
import boto3
import requests
from bs4 import BeautifulSoup
from django.views.decorators.csrf import csrf_exempt
from ultralytics import YOLO
import pandas as pd

def home(request):
    return render(request, 'home.html')

@csrf_exempt
def search(request):
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_file = request.FILES['image']  # This is an InMemoryUploadedFile or TemporaryUploadedFile
        
        image_data = uploaded_file.read()

        # If you need a PIL Image object:
        image = Image.open(BytesIO(image_data)).resize((224, 224))

        # Model configuring with ResNet50
        model = ResNet50(weights='imagenet')

        #Making Numpy Array Out of The Image
        img_array = preprocess_input(np.expand_dims(image, axis=0))

        #Calling predict functions
        predictions = model.predict(img_array)

        #Decoding Predictions
        d_prediction = decode_predictions(predictions, top=3)[0]

        # Generating Labels out of the image

        model = YOLO('yolov8n.pt')  # Downloaded automatically

        results = model(image)
        detected = set()

        for result in results:
            for box in result.boxes:
                detected.add(model.names[int(box.cls)])

        keywords = list(detected)

        print(keywords)

        # Searching Amazon Products

        API_KEY = 'fc-0091a1c8da864946a304b27241f4324b'

        url = "https://api.firecrawl.dev/v1/scrape"

        payload = {
            "url": f'https://www.flipkart.com/search?q={keywords[0].replace(" ", "%20")}&otracker=search&otracker1=search&marketplace=FLIPKART&as-show=on&as=off',
            "formats": ["json"],
            "onlyMainContent": True,
            "includeTags": [],
            "excludeTags": [],
            "headers": {},
            "waitFor": 0,
            "mobile": False,
            "skipTlsVerification": False,
            "timeout": 30000,
            "jsonOptions": {
                "schema": {},
                "systemPrompt": "",
                "prompt": ""
            },
            "actions": [
                {
                    "type": "wait",
                    "milliseconds": 8000,
                }
            ],
            "location": {
                "country": "US",
                "languages": ["en-US"]
            },
            "removeBase64Images": True,
            "blockAds": True,
            "proxy": "basic",
            "changeTrackingOptions": {
                "modes": ["git-diff"],
                "schema": {},
                "prompt": "<string>"
            }
        }
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        response = requests.request("POST", url, json=payload, headers=headers)

        print(response.text)

        products = response.json()['data']['json']['products']

        extracted_products = []
        for product in products:
            extracted_products.append({
                'title': product['name'] if 'name' in product else None,
                'image_link': product['image'] if 'image' in product else None,
                'product_url': product['url'] if 'url' in product else None,
                'price': product['price'] if 'price' in product else None,
                'ddiscount': product['discount'] if 'discount' in product else None,
            })

        return render(request, 'home.html', {'results': extracted_products})


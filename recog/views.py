import base64
import os
import pickle
import ssl

from django.core.files.base import ContentFile
from django.http import JsonResponse
from django.shortcuts import render
import face_recognition
import numpy as np
import cv2 as cv

# Create your views here.


def create_image(image, first_name, pk):
    extn, data_url = image.split(";base64,")
    img_data = base64.b64decode(data_url)
    image_file = ContentFile(img_data, name=f'{first_name}_{pk}.png')
    return image_file


def capture_cadet_image(request):

    if request.method == 'POST':
        data = request.POST
        first_name = data.get('first_name')
        number = data.get('number')
        db_path = os.path.join(os.getcwd(), "recog/db")
        data = request.POST
        image = data.get('image')

        # for recording img pickle
        image_file = create_image(image, first_name, number)
        image_pickle = np.asarray(bytearray(image_file.read()), dtype="uint8")
        image_pickle = cv.imdecode(image_pickle, cv.IMREAD_COLOR)
        embeddings = face_recognition.face_encodings(image_pickle)[0]
        file = open(os.path.join(db_path, '{}.pickle'.format(number)), 'wb')
        pickle.dump(embeddings, file)
        return JsonResponse({'response': 'Image Captured'})

    return render(request, 'recog/capture-image.html')


def take_attendance(request):

    db_path = os.path.join(os.getcwd(), "recog/db")

    if request.method == 'POST':
        try:
            img_data = request.POST.get('image')
            extn, data_url = img_data.split(";base64,")
            img_data = base64.b64decode(data_url)
            image_file = ContentFile(img_data, name=f'check_image.png')
            image = np.asarray(bytearray(image_file.read()), dtype="uint8")
            image = cv.imdecode(image, cv.IMREAD_COLOR)
            embeddings_unknown = face_recognition.face_encodings(image)

            if len(embeddings_unknown) == 0:
                resp = 'no_persons_found'
                return JsonResponse({"response": resp})
            else:
                embeddings_unknown = embeddings_unknown[0]

            db_dir = sorted(os.listdir(db_path))

            match = False
            j = 0
            while not match and j < len(db_dir):
                path_ = os.path.join(db_path, db_dir[j])

                file = open(path_, 'rb')
                embeddings = pickle.load(file)

                match = face_recognition.compare_faces([embeddings], embeddings_unknown, tolerance=0.5)[0]
                j += 1

            if match:
                resp = db_dir[j - 1]
            else:
                resp = 'unknown_person'

            return JsonResponse({"response": resp})
        except AttributeError:
            return JsonResponse({"response": 'Failed'})
    return render(request, 'recog/capture-image.html')


def home(request):
    return render(request, 'recog/home.html')


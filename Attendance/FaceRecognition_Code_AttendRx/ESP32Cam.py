import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import requests
import urllib
import csv
from datetime import date

# ESP32-CAM IP address
esp32cam_url = 'http://192.168.43.115/640x480.jpg'

rollNo={'APARNA':'2021UCA1536','PULKIT':'2021UCA1804','MUSKAN':'2021UCA1814','.DS_STORE':'0'}

# Function to fetch images from ESP32-CAM
def get_esp32cam_image():
    try:
        response = requests.get(esp32cam_url, timeout=10)
        if response.status_code == 200:
            img_array = np.array(bytearray(response.content), dtype=np.uint8)
            img = cv2.imdecode(img_array, -1)
            return img
    except Exception as e:
        print(f"Error fetching image from ESP32-CAM: {str(e)}")
    return None


path = '/Users/muskanshomef/Desktop/Attendance/FaceRecognition_Code_AttendRx/ImagesBasic'
images = []
classNames = []
mylist = os.listdir(path)
print(mylist)
for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def find_encodings(images):
    encodeList = []
    i = 0
    for img in images:
        # Check if the image is empty
        if img is None:
            print(f"Failed to read image {i + 1}/{len(images)}")
            continue
        
        # Convert the image to RGB format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Find face encodings in the image
        face_encodings = face_recognition.face_encodings(img)
        
        # Check if any face encodings are found
        if len(face_encodings) > 0:
            encode = face_encodings[0]  # Take the first face encoding
            encodeList.append(encode)
        else:
            print(f"No face found in image {i + 1}/{len(images)}")
        
        # Print progress
        print(f'Encoding {i + 1}/{len(images)} done!')
        i += 1
    return encodeList
    # encodeList = []
    # i = 0
    # for img in images:
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     encode = face_recognition.face_encodings(img)[0]
    #     encodeList.append(encode)
    #     print(f'Encoding {i}/{len(mylist)} done!')
    #     i=i+1
    # return encodeList


def markAttendance(name):
    with open('/Users/muskanshomef/Desktop/Attendance/Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{rollNo[name]},{dtString}')
    f.close()


encodelistknown = find_encodings(images)
print('Encoding Complete!')

file=open('/Users/muskanshomef/Desktop/Attendance/Attendance.csv','a')
file.writelines(f'{"ATTENDANCE for - "},{date.today()}')
file.writelines(f'\n{"NAME"},{"ROLL NUMBER"},{"TIMESTAMP"}')
file.close()

while True:
    # Capture an image from ESP32-CAM
    img = get_esp32cam_image()

    if img is not None:
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        faceCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, faceCurFrame):
            matches = face_recognition.compare_faces(encodelistknown, encodeFace)
            faceDis = face_recognition.face_distance(encodelistknown, encodeFace)
            print(faceDis)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                print(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                markAttendance(name)

        cv2.imshow('ESP32-CAM', img)
        cv2.waitKey(1)

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib


DATADIR = r"C:\Users\Sravanthi\OneDrive\Desktop\svm\PetImages"


CATEGORIES = ["Cat", "Dog"]


IMG_SIZE = 64

data = []
labels = []


for category in CATEGORIES:

    folder_path = os.path.join(DATADIR, category)
    label = CATEGORIES.index(category)

    for img in os.listdir(folder_path)[:2000]:

        try:
            img_path = os.path.join(folder_path, img)

            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

            image = image.flatten()

            data.append(image)
            labels.append(label)

        except:
            pass


data = np.array(data)
labels = np.array(labels)

print("Dataset Loaded!")


X_train, X_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.2,
    random_state=42
)


model = SVC(kernel='linear')


print("Training Started...")
model.fit(X_train, y_train)

print("Training Completed!")


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy * 100)


joblib.dump(model, "cats_dogs_svm_model.pkl")

print("Model Saved!")
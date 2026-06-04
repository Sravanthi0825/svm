

import cv2
import joblib
from tkinter import Tk
from tkinter.filedialog import askopenfilename


model = joblib.load("cats_dogs_svm_model.pkl")

IMG_SIZE = 64


Tk().withdraw()


img_path = askopenfilename(
    title="Select an Image",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
)


if not img_path:
    print("No image selected!")
    exit()


image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)


if image is None:
    print("Invalid image!")
    exit()


image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))


image = image.flatten().reshape(1, -1)


prediction = model.predict(image)


if prediction[0] == 0:
    result = "Cat Detected"
else:
    result = "Dog Detected"

print(result)


display_img = cv2.imread(img_path)

cv2.putText(
    display_img,
    result,
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
)

cv2.imshow("Prediction", display_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
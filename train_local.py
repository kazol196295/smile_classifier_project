import os
import pickle
import numpy as np
from PIL import Image
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from config import MODEL_PATH


def load_images_from_folder(folder_path: str, label: int) -> tuple[list[np.ndarray], list[int]]:
    images = []
    labels = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(valid_extensions):
            filepath = os.path.join(folder_path, filename)
            try:
                img = Image.open(filepath)
                img = img.resize((64, 64)).convert("L")
                img_array = np.array(img).flatten() / 255.0
                images.append(img_array)
                labels.append(label)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                continue

    return images, labels


def train_model(smile_folder: str, non_smile_folder: str) -> dict:
    print(f"Loading images from: {smile_folder}")
    smile_images, smile_labels = load_images_from_folder(smile_folder, label=1)
    print(f"Found {len(smile_images)} smile images")

    print(f"Loading images from: {non_smile_folder}")
    non_smile_images, non_smile_labels = load_images_from_folder(non_smile_folder, label=0)
    print(f"Found {len(non_smile_images)} non-smile images")

    X = np.array(smile_images + non_smile_images)
    y = np.array(smile_labels + non_smile_labels)

    if len(X) == 0:
        return {"success": False, "message": "No images found in the selected folders"}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training SVM model...")
    model = SVC(kernel='rbf', probability=True, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH) if os.path.dirname(MODEL_PATH) else '.', exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to: {MODEL_PATH}")

    return {
        "success": True,
        "accuracy": round(accuracy * 100, 2),
        "smile_count": len(smile_images),
        "non_smile_count": len(non_smile_images),
        "total_count": len(X),
        "message": f"Model trained successfully with {accuracy*100:.2f}% accuracy"
    }

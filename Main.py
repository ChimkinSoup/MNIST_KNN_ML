from pathlib import Path
import numpy as np

BASE_DIRECTORY = Path(__file__).parent
TEST_LABELS_FILENAME = BASE_DIRECTORY / 'data/t10k-labels.idx1-ubyte'
TEST_IMAGES_FILENAME = BASE_DIRECTORY / 'data/t10k-images.idx3-ubyte'
TRAIN_LABELS_FILENAME = BASE_DIRECTORY / 'data/train-labels.idx1-ubyte'
TRAIN_IMAGES_FILENAME = BASE_DIRECTORY / 'data/train-images.idx3-ubyte'
k_nodes = 3  # Hyperparameter - number of nearest neighbors to consider

def byte_to_int(byte):
    return int.from_bytes(byte, 'big')


def read_images(filename, max_images=None):
    if not filename.exists():
        raise FileNotFoundError(f"MNIST data file not found: {filename}")
    
    all_images = []
    try:
        with open(filename, 'rb') as file:
            _ = file.read(4)  # Magic number
            number_images = byte_to_int(file.read(4))
            if max_images:
                number_images = min(number_images, max_images)
            number_rows = byte_to_int(file.read(4))
            number_columns = byte_to_int(file.read(4))
            buffered_data = np.frombuffer(
                file.read(number_images * number_columns * number_rows), 
                dtype=np.uint8
            )
            all_images = buffered_data.reshape(number_images, number_columns * number_rows)
        return all_images.astype(np.float32)
    except Exception as e:
        raise ValueError(f"Error reading image file {filename}: {e}")


def read_labels(filename, max_labels=None):
    if not filename.exists():
        raise FileNotFoundError(f"MNIST data file not found: {filename}")
    
    all_labels = []
    try:
        with open(filename, 'rb') as file:
            _ = file.read(4)  # Magic number
            number_labels = byte_to_int(file.read(4))
            if max_labels:
                number_labels = min(number_labels, max_labels)

            for label_index in range(number_labels):
                label = byte_to_int(file.read(1))
                all_labels.append(label)
        return all_labels
    except Exception as e:
        raise ValueError(f"Error reading label file {filename}: {e}") 



def knn_singular(X_test):
    if X_test.shape != (784,):
        raise ValueError(f"Expected input shape (784,), got {X_test.shape}")
    
    try:
        X_train = read_images(TRAIN_IMAGES_FILENAME)
        y_train = read_labels(TRAIN_LABELS_FILENAME)
    except (FileNotFoundError, ValueError) as e:
        raise FileNotFoundError(
            "MNIST training data not found. Please ensure data files are in the 'data' directory."
        ) from e
    
    test_sample = X_test
    # Calculate Euclidean distance squared 
    training_distance = np.sum((X_train - test_sample) ** 2, axis=1)
    
    # Get k*10 nearest neighbors, then sort to get top k
    nearest_indices = np.argpartition(training_distance, k_nodes * 10)[:k_nodes * 10]
    nearest_indices = nearest_indices[np.argsort(training_distance[nearest_indices])]
    
    # Count occurrences of each digit in top k neighbors
    candidate_counter = {i: 0 for i in range(10)}
    
    for index in nearest_indices:
        curr_candidate = y_train[index]
        candidate_count = candidate_counter[curr_candidate] + 1
        if candidate_count == k_nodes:
            break
        else:
            candidate_counter[curr_candidate] = candidate_count
    
    return curr_candidate

if __name__ == "__main__":
    import tkinter as tk
    import Draw
    
    # Run application
    root = tk.Tk()
    app = Draw.PaintApp(root)
    root.mainloop()
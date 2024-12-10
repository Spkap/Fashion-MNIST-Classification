# Fashion MNIST Classification

This project implements Convolutional Neural Networks (CNNs) to classify Fashion MNIST dataset images using two approaches: augmented and non-augmented data training.

## Project Structure

- `Fashion MNIST without Data Augmentation.ipynb`: Basic CNN implementation
- `Fashion MNIST with Data Augmentation.ipynb`: Enhanced CNN with data augmentation
- Sample T-shirt image for testing the classification function

## Dataset

Fashion MNIST contains 70,000 grayscale images (60,000 training, 10,000 testing) across 10 fashion categories:

```python
labels = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot"
}
```

## Implementation Details

### Model Architecture

- Three convolutional blocks with batch normalization
- ReLU activation and max pooling layers
- Dropout for regularization
- Fully connected layers for classification

### Training Approaches

#### Augmented Version

Implements data augmentation with:

- Random cropping
- Horizontal flipping
- Rotation
- Color jittering

Performance:

- Validation Accuracy: 88.55%
- Training Accuracy: 85.53%
- Test Accuracy: 90.20%

#### Non-Augmented Version

Uses basic transformations:

- Resizing
- Normalization

Performance:

- Validation Accuracy: 93.50%
- Training Accuracy: 95.29%
- Test Accuracy: ~93%

## Setup and Usage

### Requirements

- Python 3.10+
- PyTorch ecosystem
- Data analysis libraries (numpy, pandas)
- Visualization tools (matplotlib, seaborn)
- Image processing (PIL)
- Machine learning utilities (scikit-learn, tqdm)

### Installation

```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn pillow tqdm
```

### Image Classification

```python
model_path = 'path/to/best_fashion_cnn.pth'
image_path = 'path/to/your/image.png'

classification = classify_image(model_path, image_path)
print(f"The image is classified as: {classification}")
```

## Performance Analysis

### Model Behavior

- Both models struggle with similar classes (Shirts vs T-shirts, Coats vs Pullovers)
- Excellent performance on distinct items (Trousers, Bags, Footwear)
- Augmented version shows better generalization despite lower raw accuracy
- Non-augmented version achieves higher accuracy but may be less robust

### Recommendation

The augmented version is recommended for production deployment due to:

- Better generalization characteristics
- Improved handling of edge cases
- More robust real-world performance
- Reduced overfitting

## Evaluation Metrics

Both versions include:

- Test accuracy measurements
- Detailed classification reports
- Confusion matrices
- Class-wise performance analysis

## Final Verdict

While the non-augmented version showed higher raw accuracy numbers, the augmented version demonstrated better generalization characteristics and real-world applicability. The slightly lower accuracy in the augmented version is a reasonable trade-off for improved model robustness and reduced overfitting. For production deployment, the augmented version would be recommended despite its slightly lower accuracy metrics.

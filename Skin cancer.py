from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

# ======================
# 1. Prepare data generators with enhanced augmentation
# ======================
train_data_generation = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,           # Rotate images randomly up to 20 degrees
    width_shift_range=0.2,       # Shift images horizontally by up to 20%
    height_shift_range=0.2,      # Shift images vertically by up to 20%
    brightness_range=[0.5, 1.5], 
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.1,        # Reserve 10% for validation
)

train_dir = r'two_types of skin cancer-20251118T153417Z-1-001\two_types of skin cancer\train'
validation_dir = train_dir
test_dir = r'two_types of skin cancer-20251118T153417Z-1-001\two_types of skin cancer\test'

# ======================
# 2. Load training data
# ======================
print("\n--- Loading Training Data ---\n")
training_set = train_data_generation.flow_from_directory(
    train_dir,
    target_size=(128, 128),      # Increased image size for better detail
    batch_size=16,
    class_mode="binary",
    subset='training',
    shuffle=True,
)

# ======================
# 3. Load validation data
# ======================
print("\n--- Loading Validation Data ---\n")
validation_set = train_data_generation.flow_from_directory(
    validation_dir,
    target_size=(128, 128),
    batch_size=16,
    class_mode="binary",
    subset='validation',
    shuffle=True,
)

# ======================
# 4. Load test data (no augmentation)
# ======================
print("\n--- Loading Test Data ---\n")
test_data_generation = ImageDataGenerator(rescale=1./255)
testing_set = test_data_generation.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=16,
    class_mode="binary",
    shuffle=False,
)

# ======================
# 5. Build improved Deep CNN model
# ======================
print("\n--- Building Improved Model ---\n")
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Flatten(),

    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification output
])

# ======================
# 6. Compile the model with Adam optimizer
# ======================
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# ======================
# 7. Train the model for 25 epochs
# ======================
print("\n--- Training Model ---\n")
history = model.fit(
    training_set,
    validation_data=validation_set,
    epochs=25
)

# ======================
# 8. Plot training and validation accuracy & loss curves
# ======================
print("\n--- Plotting Accuracy and Loss Curves ---\n")

plt.figure(figsize=(10,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# ======================
# 9. Evaluate the model on the test data and print results
# ======================
print("\n--- Evaluating Model on Test Data ---\n")
test_loss, test_acc = model.evaluate(testing_set)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print("\n==============================\n")

# Save the full model
model1.save("skin_cancer_cnn_model.h5")
print("Model saved successfully!")
import tensorflow as tf

print("Loading your existing trained model...")
# Load the model you already trained
model = tf.keras.models.load_model("models/best_model.keras")

print("Converting and exporting to SavedModel directory format...")
# Use model.export() instead of model.save() for Keras 3 directory bundles
model.export("models/best_model_savedmodel")

print("Conversion complete! You can now upload the 'models/best_model_savedmodel' folder to S3.")
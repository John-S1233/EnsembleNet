import os
import numpy as np
import tensorflow as tf
import gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from model_creation import sample_architecture, create_meta_learner, serialize_architecture

# Set parallelism options for improved stability
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

# Clear any existing sessions and collect garbage
K.clear_session()
gc.collect()

# Suppress TensorFlow warnings for cleaner output (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Directory to save all models
BASE_SAVE_DIR = 'models'

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

# -------------------- Encoding Setup -------------------- #
MODEL_TYPES = ["Transformer", "CNN", "DenseNet", "Inception", "MobileNet", "LNN", "AttentionCNN", "RNN", "Hybrid", "GAN"]
ACTIVATIONS = ['relu', 'tanh', 'elu', 'leaky_relu']

# Initialize encoders/scalers
model_type_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
activation_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
model_type_encoder.fit(np.array(MODEL_TYPES).reshape(-1, 1))
activation_encoder.fit(np.array(ACTIVATIONS).reshape(-1, 1))
numerical_scaler = StandardScaler()

def encode_hyperparameters(params):
    """Encode hyperparameters into a numerical vector."""
    encoded = []
    if 'activation' in params:
        encoded_activation = activation_encoder.transform(np.array([params['activation']]).reshape(-1, 1))
        encoded.append(encoded_activation.flatten())
    
    numerical_features = []
    for key, value in params.items():
        if key not in ['activation'] and isinstance(value, (int, float, list, tuple)):
            if isinstance(value, list) or isinstance(value, tuple):
                numerical_features.extend([item for item in value if isinstance(item, (int, float))])
            elif isinstance(value, (int, float)):
                numerical_features.append(value)
    if numerical_features:
        numerical_scaler.partial_fit(np.array(numerical_features).reshape(-1, 1))
        numerical_scaled = numerical_scaler.transform(np.array(numerical_features).reshape(-1, 1)).flatten()
        encoded.append(numerical_scaled)
    
    if encoded:
        return np.concatenate(encoded)
    else:
        return np.array([])

# -------------------- Data Loading Function for CIFAR-100 -------------------- #
def load_and_preprocess_data():
    """Load and preprocess the CIFAR-100 dataset from a custom directory."""
    # Specify your custom dataset path
    dataset_dir = os.path.join(os.getcwd(), 'datasets')  # Change 'datasets' to your preferred folder name
    dataset_tar_gz = os.path.join(dataset_dir, 'cifar-100-python.tar.gz')

    # Create the dataset directory if it doesn't exist
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # Check if the dataset is already downloaded and extracted
    extracted_dir = os.path.join(dataset_dir, 'cifar-100-python')
    if not os.path.exists(extracted_dir):
        print("Dataset not found locally. Downloading CIFAR-100 dataset...")
        dataset_tar_gz = tf.keras.utils.get_file(
            'cifar-100-python.tar.gz',
            origin='https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz',
            cache_dir=dataset_dir,
            cache_subdir='',
            extract=True
        )
        print("Download completed. Dataset extracted.")
    else:
        print("Dataset already exists locally.")

    # Load dataset
    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

    # Normalize the pixel values to be between 0 and 1
    x_train_full = x_train_full.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Split the full training set into training and validation sets
    from sklearn.model_selection import train_test_split
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size=0.1, random_state=42, stratify=y_train_full
    )

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

# -------------------- Main Training Loop -------------------- #
def main():
    # Load and preprocess data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data()

    # Configuration
    HyperEpochs = 50          # Number of times to repeat the process
    num_models = 10            # Number of sub-models
    batch_size = 128           # Batch size for training
    sub_model_epochs = 15      # Number of epochs to train each sub-model
    meta_learner_epochs = 45  # Number of epochs to train the meta-learner per HyperEpoch

    # Ensure base save directory exists
    ensure_dir(BASE_SAVE_DIR)

    # Initialize sub-models and meta-learner
    sub_models = []
    sub_model_archs = []
    sub_model_hyperparams = []

    for model_idx in range(num_models):
        model_path = os.path.join(BASE_SAVE_DIR, f"sub_model_{model_idx+1}.h5")
        if os.path.exists(model_path):
            try:
                model, arch_type, params = sample_architecture(None)
                model = tf.keras.models.load_model(model_path)
                print(f"Loaded sub-model {model_idx+1}/{num_models} from disk.")
            except Exception as e:
                print(f"Error loading sub-model {model_idx+1}: {e}")
                print("Initializing a new sub-model.")
                model, arch_type, params = sample_architecture(None)
                optimizer = Adam(learning_rate=0.001)
                model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        else:
            model, arch_type, params = sample_architecture(None)
            arch_string = serialize_architecture(arch_type, params)
            print(f"Initialized sub-model {model_idx+1}/{num_models}: {arch_string}")

            optimizer = Adam(learning_rate=0.001)
            model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        sub_models.append(model)
        sub_model_archs.append(serialize_architecture(arch_type, params))
        sub_model_hyperparams.append(params)

    encoded_hyperparams = [encode_hyperparameters(params) for params in sub_model_hyperparams]
    stacked_hyperparams = np.concatenate(encoded_hyperparams)

    # Prepare meta-learner
    predictions_shape = (100 * num_models,)
    hyperparams_shape = (stacked_hyperparams.shape[0],)
    x_val_shape = (32 * 32 * 3,)

    meta_learner_path = os.path.join(BASE_SAVE_DIR, "meta_learner.h5")
    if os.path.exists(meta_learner_path):
        try:
            meta_learner = tf.keras.models.load_model(meta_learner_path)
            print("Loaded meta-learner from disk.")
        except Exception as e:
            print(f"Error loading meta-learner: {e}")
            meta_learner = create_meta_learner(predictions_shape, hyperparams_shape, x_val_shape)
            meta_learner.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    else:
        meta_learner = create_meta_learner(predictions_shape, hyperparams_shape, x_val_shape)
        meta_learner.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    sub_model_checkpoints = []
    for model_idx in range(num_models):
        checkpoint_path = os.path.join(BASE_SAVE_DIR, f"sub_model_{model_idx+1}.h5")
        checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
        sub_model_checkpoints.append(checkpoint)

    meta_learner_checkpoint = ModelCheckpoint(filepath=meta_learner_path, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)

    for epoch in range(1, HyperEpochs + 1):
        print(f"\n=== HyperEpoch {epoch}/{HyperEpochs} ===")

        indices = np.arange(x_train.shape[0])
        np.random.shuffle(indices)
        x_train_shuffled = x_train[indices]
        y_train_shuffled = y_train[indices]

        for model_idx, model in enumerate(sub_models):
            print(f"Training sub-model {model_idx+1}/{num_models} for {sub_model_epochs} epochs")
            model.fit(x_train_shuffled, y_train_shuffled, batch_size=batch_size, epochs=sub_model_epochs, validation_data=(x_val, y_val), callbacks=[sub_model_checkpoints[model_idx]])

            K.clear_session()
            del model
            gc.collect()

            model = tf.keras.models.load_model(os.path.join(BASE_SAVE_DIR, f"sub_model_{model_idx+1}.h5"))
            sub_models[model_idx] = model

        print("Generating predictions for meta-learner.")
        sub_model_predictions_val = [model.predict(x_val, batch_size=batch_size) for model in sub_models]
        sub_model_predictions_test = [model.predict(x_test, batch_size=batch_size) for model in sub_models]

        stacked_val_preds = np.concatenate(sub_model_predictions_val, axis=1)
        stacked_test_preds = np.concatenate(sub_model_predictions_test, axis=1)

        x_val_flat = x_val.reshape((x_val.shape[0], -1))
        x_test_flat = x_test.reshape((x_test.shape[0], -1))

        hyperparams_input_val = np.tile(stacked_hyperparams, (stacked_val_preds.shape[0], 1))
        hyperparams_input_test = np.tile(stacked_hyperparams, (stacked_test_preds.shape[0], 1))

        print(f"Training meta-learner for {meta_learner_epochs} epochs")
        meta_learner.fit([stacked_val_preds, hyperparams_input_val, x_val_flat], y_val, batch_size=batch_size, epochs=meta_learner_epochs, validation_data=([stacked_test_preds, hyperparams_input_test, x_test_flat], y_test), callbacks=[meta_learner_checkpoint])

        print("Evaluating meta-learner.")
        meta_test_preds = meta_learner.predict([stacked_test_preds, hyperparams_input_test, x_test_flat], batch_size=batch_size)
        test_accuracy = np.mean(np.argmax(meta_test_preds, axis=1) == y_test)
        print(f"Meta-learner Test Accuracy after HyperEpoch {epoch}: {test_accuracy:.4f}")

        K.clear_session()
        del meta_test_preds
        gc.collect()

    print("Training completed.")
    meta_test_preds = meta_learner.predict([stacked_test_preds, hyperparams_input_test, x_test_flat], batch_size=batch_size)
    final_test_accuracy = np.mean(np.argmax(meta_test_preds, axis=1) == y_test)
    print(f"Final Meta-Learner Test Accuracy: {final_test_accuracy:.4f}")

if __name__ == "__main__":
    main()

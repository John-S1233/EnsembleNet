import os
import numpy as np
import tensorflow as tf
import gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model_creation import sample_architecture, create_multibranch_meta_learner, serialize_architecture

# Set parallelism options 
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

K.clear_session()
gc.collect()

# Directory to save all models
BASE_SAVE_DIR = 'models'

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

# -------------------- Encoding Setup -------------------- #

# Define possible categories based on your model_creation.py
MODEL_TYPES = ["Transformer", "CNN", "DenseNet", "Inception", "MobileNet",
               "LNN", "AttentionCNN", "RNN", "Hybrid", "GAN"]

ACTIVATIONS = ['relu', 'tanh', 'elu', 'leaky_relu']

# Initialize encoders/scalers with updated parameter name
model_type_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
activation_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

# Fit encoders with all possible categories
model_type_encoder.fit(np.array(MODEL_TYPES).reshape(-1, 1))
activation_encoder.fit(np.array(ACTIVATIONS).reshape(-1, 1))

# Initialize scaler once for numerical features
numerical_scaler = StandardScaler()

def encode_hyperparameters(params, arch_type):
    """Encode hyperparameters and architecture type into a numerical vector."""
    encoded = []
    
    if arch_type in MODEL_TYPES:
        encoded_arch = model_type_encoder.transform(np.array([arch_type]).reshape(-1, 1))
        encoded.append(encoded_arch.flatten())
    else:
        encoded_arch = model_type_encoder.transform(np.array(['Unknown']).reshape(-1, 1))
        encoded.append(encoded_arch.flatten())
    
    if 'activation' in params:
        encoded_activation = activation_encoder.transform(np.array([params['activation']]).reshape(-1, 1))
        encoded.append(encoded_activation.flatten())
    
    # Encode numerical hyperparameters (e.g., number of layers)
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

# -------------------- Data Loading Function -------------------- #
def load_and_preprocess_data():
    """Load and preprocess the CIFAR-100 dataset."""
    from tensorflow.keras.datasets import cifar100

    # Load CIFAR-100 data
    (x_train_full, y_train_full), (x_test, y_test) = cifar100.load_data()
    
    # Normalize pixel values
    x_train_full = x_train_full.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    y_train_full = y_train_full.astype(np.int32).reshape(-1)
    y_test = y_test.astype(np.int32).reshape(-1)

    # Split the training set 
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size=0.1, random_state=42, stratify=y_train_full
    )

    x_meta_train, x_meta_val, y_meta_train, y_meta_val = train_test_split(
        x_val, y_val, test_size=0.5, random_state=42, stratify=y_val
    )

    # Print unique labels to verify correctness
    print("Unique y_train labels:", np.unique(y_train))
    print("Unique y_meta_train labels:", np.unique(y_meta_train))
    print("Unique y_meta_val labels:", np.unique(y_meta_val))
    print("Unique y_test labels:", np.unique(y_test))

    return (x_train, y_train), (x_meta_train, y_meta_train), (x_meta_val, y_meta_val), (x_test, y_test)

# -------------------- Main Training Loop -------------------- #
def main():
    # Load and preprocess data
    (x_train, y_train), (x_meta_train, y_meta_train), (x_meta_val, y_meta_val), (x_test, y_test) = load_and_preprocess_data()

    # Configuration
    HyperEpochs = 50              # Number of times to repeat the process
    num_models = 10                # Number of sub-models
    batch_size = 128              # Batch size for training
    sub_model_epochs = 25         # Number of epochs to train each sub-model
    meta_learner_epochs = 25      # Number of epochs to train the meta-learner per HyperEpoch
    meta_learner_batch_size = 32  # Batch size for meta learner

    ensure_dir(BASE_SAVE_DIR)

    # Initialize sub-models and meta-learner
    sub_models = []
    sub_model_archs = []
    sub_model_hyperparams = []

    for model_idx in range(num_models):
        model_path = os.path.join(BASE_SAVE_DIR, f"sub_model_{model_idx+1}.h5")
        if os.path.exists(model_path):
            try:
                model = tf.keras.models.load_model(model_path)
                print(f"Loaded sub-model {model_idx+1}/{num_models} from disk.")
            except Exception as e:
                print(f"Error loading sub-model {model_idx+1}: {e}")
                print("Initializing a new sub-model.")
                model, arch_type, params = sample_architecture(None)
               
                # Compile the sub-model
                optimizer = Adam(learning_rate=0.001)
                model.compile(
                    optimizer=optimizer,
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                sub_model_archs.append(serialize_architecture(arch_type, params))
                sub_model_hyperparams.append((params, arch_type))
        else:
            model, arch_type, params = sample_architecture(None) 
            arch_string = serialize_architecture(arch_type, params) 
            print(f"Initialized sub-model {model_idx+1}/{num_models}: {arch_string}")

            # Compile the sub-model
            optimizer = Adam(learning_rate=0.001)
            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            sub_model_archs.append(arch_string)
            sub_model_hyperparams.append((params, arch_type))
        sub_models.append(model)

    # Encode hyperparameters 
    encoded_hyperparams = [encode_hyperparameters(params, arch_type) for (params, arch_type) in sub_model_hyperparams]
    stacked_hyperparams = np.concatenate(encoded_hyperparams) 

    # Prepare meta-learner
    predictions_shape = (100 * num_models,)  
    image_shape = x_meta_val.shape[1:]      

    meta_learner_path = os.path.join(BASE_SAVE_DIR, "meta_learner.h5")
    if os.path.exists(meta_learner_path):
        try:
            meta_learner = tf.keras.models.load_model(meta_learner_path)
            print("Loaded meta-learner from disk.")
        except Exception as e:
            print(f"Error loading meta-learner: {e}")
            print("Initializing a new meta-learner.")
            meta_learner = create_multibranch_meta_learner(predictions_shape, stacked_hyperparams.shape[0], image_shape)
            meta_learner.compile(
                optimizer=Adam(learning_rate=0.0001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
    else:
        # Initialize a new meta-learner with image data
        meta_learner = create_multibranch_meta_learner(predictions_shape, stacked_hyperparams.shape[0], image_shape)
        print("Initialized meta-learner.")

        meta_learner.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    # Define ModelCheckpoint callbacks for sub-models
    sub_model_checkpoints = []
    for model_idx in range(num_models):
        checkpoint_path = os.path.join(BASE_SAVE_DIR, f"sub_model_{model_idx+1}.h5")
        checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=False, 
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        )
        sub_model_checkpoints.append(checkpoint)

    # Define ModelCheckpoint callback for meta-learner
    meta_learner_checkpoint = ModelCheckpoint(
        filepath=meta_learner_path,
        save_weights_only=False, 
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )

    # Define EarlyStopping callback for meta-learner to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=25,  # Adjusted patience for earlier stopping
        restore_best_weights=True,
        verbose=1
    )

    # Main training loop
    for hyper_epoch in range(1, HyperEpochs + 1):
        print(f"\n=== HyperEpoch {hyper_epoch}/{HyperEpochs} ===")

        # Shuffle training data
        indices = np.arange(x_train.shape[0])
        np.random.shuffle(indices)
        x_train_shuffled = x_train[indices]
        y_train_shuffled = y_train[indices]

        # Train each sub-model
        for model_idx, model in enumerate(sub_models):
            print(f"Training sub-model {model_idx+1}/{num_models} for {sub_model_epochs} epochs")
            history = model.fit(
                x_train_shuffled,
                y_train_shuffled,
                batch_size=batch_size,
                epochs=sub_model_epochs,
                verbose=1,
                validation_data=(x_meta_train, y_meta_train),
                callbacks=[sub_model_checkpoints[model_idx]]
            )

            print(f"Sub-model {model_idx+1} Training History: {history.history}")

            loss, acc = model.evaluate(x_meta_val, y_meta_val, verbose=0)
            print(f"Sub-model {model_idx+1} Validation Accuracy: {acc:.4f}")

            print(f"Cleaning up memory after training sub-model {model_idx+1}/{num_models}")
            K.clear_session()
            del model
            gc.collect()

            model = tf.keras.models.load_model(os.path.join(BASE_SAVE_DIR, f"sub_model_{model_idx+1}.h5"))
            sub_models[model_idx] = model  

        # Generate predictions for meta-learner
        print("Generating predictions for meta-learner.")
        sub_model_predictions_meta_train = [model.predict(x_meta_train, batch_size=meta_learner_batch_size) for model in sub_models]
        stacked_meta_train_preds = np.concatenate(sub_model_predictions_meta_train, axis=1)

        sub_model_predictions_meta_val = [model.predict(x_meta_val, batch_size=meta_learner_batch_size) for model in sub_models]
        stacked_meta_val_preds = np.concatenate(sub_model_predictions_meta_val, axis=1)

        # Prepare hyperparameters input
        hyperparams_input_meta_train = np.tile(stacked_hyperparams, (stacked_meta_train_preds.shape[0], 1))
        hyperparams_input_meta_val = np.tile(stacked_hyperparams, (stacked_meta_val_preds.shape[0], 1))

        # Train meta-learner 
        print(f"Training meta-learner for {meta_learner_epochs} epochs")
        meta_history = meta_learner.fit(
            [stacked_meta_train_preds, hyperparams_input_meta_train, x_meta_train],
            y_meta_train,
            batch_size=meta_learner_batch_size,
            epochs=meta_learner_epochs,
            verbose=1,
            validation_data=(
                [stacked_meta_val_preds, hyperparams_input_meta_val, x_meta_val],
                y_meta_val
            ),
            callbacks=[meta_learner_checkpoint, early_stopping]
        )

        print(f"Meta-Learner Training History: {meta_history.history}")

        # Evaluate meta-learner on the test set
        print("Evaluating meta-learner.")
        sub_model_predictions_test = [model.predict(x_test, batch_size=batch_size) for model in sub_models]
        stacked_test_preds = np.concatenate(sub_model_predictions_test, axis=1)

        hyperparams_input_test = np.tile(stacked_hyperparams, (stacked_test_preds.shape[0], 1))

        # Make predictions with the meta-learner
        meta_test_preds = meta_learner.predict(
            [stacked_test_preds, hyperparams_input_test, x_test],
            batch_size=batch_size
        )
        test_accuracy = np.mean(np.argmax(meta_test_preds, axis=1) == y_test)
        print(f"Meta-learner Test Accuracy after HyperEpoch {hyper_epoch}: {test_accuracy:.4f}")

        print("Cleaning up memory after meta-learner training")
        K.clear_session()
        del meta_test_preds
        gc.collect()

    print("Training completed.")
    print("\n=== Final Evaluation on Test Set ===")
    
    # Final Evaluation
    sub_model_predictions_test = [model.predict(x_test, batch_size=batch_size) for model in sub_models]
    stacked_test_preds = np.concatenate(sub_model_predictions_test, axis=1)

    hyperparams_input_test = np.tile(stacked_hyperparams, (stacked_test_preds.shape[0], 1))

    # Make predictions with the meta-learner
    meta_test_preds = meta_learner.predict(
        [stacked_test_preds, hyperparams_input_test, x_test],
        batch_size=batch_size
    )
    final_test_accuracy = np.mean(np.argmax(meta_test_preds, axis=1) == y_test)
    print(f"Final Meta-Learner Test Accuracy: {final_test_accuracy:.4f}")

if __name__ == "__main__":
    main()

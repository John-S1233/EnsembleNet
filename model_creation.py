import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

# Random hyperparameter function
def random_choice(param_list):
    """Randomly select an item from a list."""
    return random.choice(param_list)

def serialize_architecture(arch_type, params):
    """Serialize architecture type and parameters into a unique string."""
    return f"{arch_type}-{params}"

# -------------------- Model Creation Functions -------------------- #

def create_multibranch_meta_learner(predictions_shape, hyperparams_dim, image_shape, num_classes=100):
    """
    Creates a multibranch meta-learner model.

    Args:
        predictions_shape (tuple): Shape of the concatenated sub-model predictions (e.g., (500,)).
        hyperparams_dim (int): Dimensionality of the encoded hyperparameters vector.
        image_shape (tuple): Shape of the CIFAR-100 image data (e.g., (32, 32, 3)).
        num_classes (int): Number of output classes (default is 100 for CIFAR-100).

    Returns:
        keras.Model: Compiled multibranch meta-learner model.
    """
    # -------------------- Sub-Model Predictions Branch -------------------- #
    predictions_input = layers.Input(shape=predictions_shape, name='predictions_input')
    x_pred = layers.Dense(256, activation='relu')(predictions_input)
    x_pred = layers.BatchNormalization()(x_pred)
    x_pred = layers.Dropout(0.5)(x_pred)
    x_pred = layers.Dense(128, activation='relu')(x_pred)
    x_pred = layers.BatchNormalization()(x_pred)
    x_pred = layers.Dropout(0.5)(x_pred)

    # -------------------- Hyperparameters Branch -------------------- #
    hyperparams_input = layers.Input(shape=(hyperparams_dim,), name='hyperparams_input')
    x_hyper = layers.Dense(128, activation='relu')(hyperparams_input)
    x_hyper = layers.BatchNormalization()(x_hyper)
    x_hyper = layers.Dropout(0.5)(x_hyper)
    x_hyper = layers.Dense(64, activation='relu')(x_hyper)
    x_hyper = layers.BatchNormalization()(x_hyper)
    x_hyper = layers.Dropout(0.5)(x_hyper)

    # -------------------- CIFAR-100 Images Branch -------------------- #
    images_input = layers.Input(shape=image_shape, name='images_input')
    x_img = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(images_input)
    x_img = layers.BatchNormalization()(x_img)
    x_img = layers.MaxPooling2D((2, 2))(x_img)
    x_img = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x_img)
    x_img = layers.BatchNormalization()(x_img)
    x_img = layers.MaxPooling2D((2, 2))(x_img)
    x_img = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x_img)
    x_img = layers.BatchNormalization()(x_img)
    x_img = layers.GlobalAveragePooling2D()(x_img)
    x_img = layers.Dense(128, activation='relu')(x_img)
    x_img = layers.BatchNormalization()(x_img)
    x_img = layers.Dropout(0.5)(x_img)

    # -------------------- Concatenation and Fusion -------------------- #
    concatenated = layers.concatenate([x_pred, x_hyper, x_img], name='concatenated_features')
    x = layers.Dense(256, activation='relu')(concatenated)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    # -------------------- Output Layer -------------------- #
    output = layers.Dense(num_classes, activation='softmax', name='output')(x)

    # -------------------- Model Creation -------------------- #
    model = models.Model(inputs=[predictions_input, hyperparams_input, images_input], outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def create_transformer_model():
    """Create a simplified Transformer-based model for CIFAR-100."""
    input_layer = layers.Input(shape=(32, 32, 3)) 

    x = layers.Dense(64, activation='relu')(input_layer)  

    # Transformer Encoder Layer
    transformer_block = layers.MultiHeadAttention(num_heads=2, key_dim=64)
    attn_output = transformer_block(x, x)  # Self-attention
    x = layers.Add()([x, attn_output])  # Residual connection
    x = layers.LayerNormalization()(x)

    # Feed-Forward Network
    ff = layers.Dense(128, activation='relu')(x)
    ff = layers.Dense(64, activation='relu')(ff)
    x = layers.Add()([x, ff])  # Residual connection
    x = layers.LayerNormalization()(x)

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x) 

    # Classification Head
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(100, activation='softmax')(x) 

    model = models.Model(inputs=input_layer, outputs=output)
    params = {
        "num_heads": 2,
        "key_dim": 64,
        "dense_units": [128, 64],
        "dropout_rate": 0.3
    }
    return model, params

def create_cnn_model():
    num_conv_layers = random_choice([1, 2, 3, 4])
    num_filters = [random_choice([32, 64, 128, 256]) for _ in range(num_conv_layers)]
    kernel_size = random_choice([(3, 3), (5, 5)])
    activation = random_choice(['relu', 'tanh', 'elu'])

    model = tf.keras.Sequential()
    model.add(layers.InputLayer(input_shape=(32, 32, 3)))

    for filters in num_filters:
        model.add(layers.Conv2D(filters, kernel_size=kernel_size, activation=activation, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(random_choice([64, 128, 256]), activation=activation))
    model.add(layers.Dropout(random_choice([0.3, 0.4, 0.5])))
    model.add(layers.Dense(100, activation='softmax'))

    params = {"num_conv_layers": num_conv_layers, "num_filters": num_filters, "kernel_size": kernel_size, "activation": activation}
    return model, params

def create_densenet_model():
    growth_rate = random_choice([12, 24, 32])
    num_layers_per_block = random_choice([4, 6, 8])
    activation = random_choice(['relu', 'leaky_relu', 'elu'])

    input_layer = layers.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, (3, 3), activation=activation, padding='same')(input_layer)
    x = layers.BatchNormalization()(x)

    for _ in range(num_layers_per_block):
        cb = layers.BatchNormalization()(x)
        cb = layers.Activation(activation)(cb)
        cb = layers.Conv2D(growth_rate, (3, 3), padding='same')(cb)
        x = layers.concatenate([x, cb])

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(random_choice([128, 256]), activation=activation)(x)
    x = layers.Dropout(random_choice([0.3, 0.4, 0.5]))(x)
    output = layers.Dense(100, activation='softmax')(x)

    model = models.Model(inputs=input_layer, outputs=output)
    params = {"growth_rate": growth_rate, "num_layers_per_block": num_layers_per_block, "activation": activation}
    return model, params

def create_inception_model():
    activation = random_choice(['relu', 'leaky_relu', 'elu'])
    num_inception_blocks = random_choice([2, 3])

    input_layer = layers.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, (1, 1), activation=activation, padding='same')(input_layer)
    x = layers.Conv2D(64, (3, 3), activation=activation, padding='same')(x)

    for _ in range(num_inception_blocks):
        branch1 = layers.Conv2D(32, (1, 1), activation=activation, padding='same')(x)
        branch1 = layers.Conv2D(32, (3, 3), activation=activation, padding='same')(branch1)

        branch2 = layers.Conv2D(32, (1, 1), activation=activation, padding='same')(x)
        branch2 = layers.Conv2D(32, (5, 5), activation=activation, padding='same')(branch2)

        branch3 = layers.MaxPooling2D(pool_size=(3, 3), strides=(1,1), padding='same')(x)
        branch3 = layers.Conv2D(32, (1, 1), activation=activation, padding='same')(branch3)

        x = layers.concatenate([branch1, branch2, branch3], axis=-1)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(random_choice([128, 256]), activation=activation)(x)
    x = layers.Dropout(random_choice([0.3, 0.4, 0.5]))(x)
    output = layers.Dense(100, activation='softmax')(x)

    model = models.Model(inputs=input_layer, outputs=output)
    params = {"num_inception_blocks": num_inception_blocks, "activation": activation}
    return model, params

def create_mobilenet_model():
    alpha = random_choice([0.35, 0.50, 0.75, 1.0])
    depth_multiplier = random_choice([1, 2])
    activation = random_choice(['relu', 'leaky_relu', 'elu'])

    model = tf.keras.Sequential()
    model.add(layers.InputLayer(input_shape=(32, 32, 3)))

    model.add(layers.Conv2D(int(32 * alpha), (3, 3), padding='same', activation=activation))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    for filters in [64, 128, 256]:
        model.add(layers.DepthwiseConv2D((3, 3), padding='same', activation=activation, depth_multiplier=depth_multiplier))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(int(filters * alpha), (1, 1), padding='same', activation=activation))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(random_choice([128, 256]), activation=activation))
    model.add(layers.Dropout(random_choice([0.3, 0.4, 0.5])))
    model.add(layers.Dense(100, activation='softmax'))

    params = {"alpha": alpha, "depth_multiplier": depth_multiplier, "activation": activation}
    return model, params

def create_lnn_model():
    """Create a simplified Liquid Neural Network model for CIFAR-100."""
    num_units = random_choice([64, 128, 256])
    activation = random_choice(['tanh', 'relu'])
    return_sequences = random_choice([False, True])

    input_layer = layers.Input(shape=(32, 32, 3)) 

    x = layers.Reshape((32, 32 * 3))(input_layer)

    # RNN Layer
    x = layers.SimpleRNN(num_units, activation=activation, return_sequences=return_sequences)(x)

    if return_sequences:
        x = layers.GlobalAveragePooling1D()(x)
    else:
        x = layers.BatchNormalization()(x)

    # Dense layers
    x = layers.Dense(random_choice([128, 256]), activation='relu')(x)
    x = layers.Dropout(random_choice([0.3, 0.4, 0.5]))(x)
    output = layers.Dense(100, activation='softmax')(x) 

    model = models.Model(inputs=input_layer, outputs=output)
    params = {
        "num_units": num_units,
        "activation": activation,
        "return_sequences": return_sequences
    }
    return model, params

def create_attention_augmented_cnn():
    """Create an Attention-Augmented CNN model for CIFAR-100."""
    activation = random_choice(['relu', 'tanh', 'elu'])
    num_conv_layers = random_choice([1, 2, 3])
    num_filters = [random_choice([32, 64, 128]) for _ in range(num_conv_layers)]

    input_layer = layers.Input(shape=(32, 32, 3))
    x = input_layer

    for filters in num_filters:
        x = layers.Conv2D(filters, (3, 3), activation=activation, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Reshape((x.shape[1] * x.shape[2], x.shape[3]))(x)
    attention = layers.MultiHeadAttention(num_heads=2, key_dim=32)(x, x)
    x = layers.Add()([x, attention])
    x = layers.LayerNormalization()(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(random_choice([128, 256]), activation=activation)(x)
    x = layers.Dropout(random_choice([0.3, 0.4, 0.5]))(x)
    output = layers.Dense(100, activation='softmax')(x)

    model = models.Model(inputs=input_layer, outputs=output)
    params = {"num_conv_layers": num_conv_layers, "num_filters": num_filters, "activation": activation, "num_heads": 2, "key_dim": 32}
    return model, params

def create_rnn_model():
    """Create a simplified RNN-based model for CIFAR-100."""
    rnn_type = random_choice(['SimpleRNN', 'GRU', 'LSTM'])
    units = random_choice([64, 128, 256])
    activation = random_choice(['tanh', 'relu'])
    return_sequences = random_choice([False, True])


    input_layer = layers.Input(shape=(32, 32, 3)) 


    x = layers.Reshape((32, 32 * 3))(input_layer)

    # RNN Layer
    if rnn_type == 'SimpleRNN':
        x = layers.SimpleRNN(units, activation=activation, return_sequences=return_sequences)(x)
    elif rnn_type == 'GRU':
        x = layers.GRU(units, activation=activation, return_sequences=return_sequences)(x)
    elif rnn_type == 'LSTM':
        x = layers.LSTM(units, activation=activation, return_sequences=return_sequences)(x)

    if return_sequences:
        x = layers.GlobalAveragePooling1D()(x)
    else:
        x = layers.BatchNormalization()(x)

    # Dense Layers
    x = layers.Dense(random_choice([128, 256]), activation='relu')(x)
    x = layers.Dropout(random_choice([0.3, 0.4, 0.5]))(x)
    output = layers.Dense(100, activation='softmax')(x)  

    model = models.Model(inputs=input_layer, outputs=output)
    params = {
        "rnn_type": rnn_type,
        "units": units,
        "activation": activation,
        "return_sequences": return_sequences
    }
    return model, params

def create_hybrid_model():
    """Create a hybrid model combining CNN and RNN layers for CIFAR-100."""
    activation = random_choice(['relu', 'tanh', 'elu'])
    num_conv_layers = random_choice([2, 3])
    num_rnn_units = random_choice([64, 128])

    input_layer = layers.Input(shape=(32, 32, 3))
    x = input_layer

    for _ in range(num_conv_layers):
        x = layers.Conv2D(random_choice([32, 64]), (3,3), activation=activation, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2,2))(x)

    x = layers.Reshape((x.shape[1], x.shape[2]*x.shape[3]))(x)
    x = layers.GRU(num_rnn_units, activation=activation)(x)

    x = layers.Dense(random_choice([128, 256]), activation=activation)(x)
    x = layers.Dropout(random_choice([0.3, 0.4, 0.5]))(x)
    output = layers.Dense(100, activation='softmax')(x)

    model = models.Model(inputs=input_layer, outputs=output)
    params = {"num_conv_layers": num_conv_layers, "num_rnn_units": num_rnn_units, "activation": activation}
    return model, params

def create_gan_model():
    """Create a simplified GAN model for CIFAR-100, adapted for classification."""
    from tensorflow.keras.models import Model
    from tensorflow.keras import Input

    def build_generator():
        noise = Input(shape=(100,))
        x = layers.Dense(256, activation='relu')(noise)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(32 * 32 * 3, activation='tanh')(x)
        x = layers.Reshape((32, 32, 3))(x)
        generator = Model(noise, x, name='Generator')
        return generator

    def build_discriminator():
        input_image = Input(shape=(32, 32, 3))
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_image)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)

        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)

        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(100, activation='softmax')(x)

        discriminator = Model(input_image, output, name='Discriminator')
        return discriminator

    generator = build_generator()
    discriminator = build_discriminator()

    model = discriminator
    params = {
        "gan_type": "StandardGAN",
        "generator_layers": [100, 256, 512, 1024, 32 * 32 * 3],
        "discriminator_layers": [64, 128, 256],
        "activation": "relu",
        "optimizer": "Adam",
        "loss": "sparse_categorical_crossentropy"
    }
    return model, params

# -------------------- Sampling Function -------------------- #

def sample_architecture(adjacency_matrix=None):
    """Randomly sample a model architecture."""
    arch_type = random_choice([
        "Transformer",
        "CNN",
        "DenseNet",
        "Inception",
        "MobileNet",
        "LNN",
        "AttentionCNN",
        "RNN",
        "Hybrid",
        "GAN"
    ])
    
    if arch_type == "Transformer":
        model, params = create_transformer_model()
    elif arch_type == "CNN":
        model, params = create_cnn_model()
    elif arch_type == "DenseNet":
        model, params = create_densenet_model()
    elif arch_type == "Inception":
        model, params = create_inception_model()
    elif arch_type == "MobileNet":
        model, params = create_mobilenet_model()
    elif arch_type == "LNN":
        model, params = create_lnn_model()
    elif arch_type == "AttentionCNN":
        model, params = create_attention_augmented_cnn()
    elif arch_type == "RNN":
        model, params = create_rnn_model()
    elif arch_type == "Hybrid":
        model, params = create_hybrid_model()
    elif arch_type == "GAN":
        model, params = create_gan_model()
    else:
        raise ValueError(f"Unsupported architecture type: {arch_type}")
    
    arch_string = serialize_architecture(arch_type, params)
    print(f"Initializing {arch_type} model with parameters: {params}")
    return model, arch_type, params

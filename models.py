"""Used model architectures and helper functions."""

import tensorflow as tf
from tensorflow.keras import layers


def data_augmentation(img_in: tf.Tensor, stateless_seed: tuple[int, int] = None):
    """Data augmentation fucntion"""
    img = tf.image.stateless_random_flip_up_down(img_in, seed=stateless_seed)
    img = tf.image.stateless_random_flip_left_right(img, seed=stateless_seed)
    img = tf.image.stateless_random_brightness(img, max_delta=0.15, seed=stateless_seed)
    img = tf.image.stateless_random_contrast(
        img, lower=0.1, upper=0.9, seed=stateless_seed
    )
    return img


def mlp(x_0: tf.Tensor, hidden_units: tuple[int, int], dropout_rate: float):
    """Function to implement MLP in ViT."""
    assert dropout_rate < 1, "Dropout rate has to be in range [0,1]."
    for units in hidden_units:
        x_0 = layers.Dense(
            units,
            activation=tf.nn.gelu,
        )(x_0)
        x_0 = layers.Dropout(dropout_rate)(x_0)
    return x_0


class PatchEncoder(layers.Layer):
    """A layer to create patch embdeddings from input image when using ViT."""

    def __init__(self, num_patches: int, projection_dim: int):
        super().__init__()
        self.projection_dim = projection_dim
        self.num_patches = num_patches
        self.projection = layers.Dense(units=self.projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=self.projection_dim
        )

    def call(self, patch):
        """Call method for Patch encoder"""
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def create_cnn_network(
    input_shape: tuple[int, int, int] = (224, 224, 3), num_classes: int = 3
):
    """Function to create model using the combined achitecture of pre-trained CNN with
    standard dense layer classifier. Default is EfficientNetV2S. Backbone layers are
    frozen by default."""
    base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2S(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )

    base_model.trainable = False

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Reshape((input_shape[0], input_shape[1])),
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1024),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(512),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


def create_cnn_vit_network(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 3,
    projection_dim: int = 64,
    transformer_layers: int = 6,
    num_heads: int = 6,
    mlp_head_units: tuple[int, int] = [2048, 1024],
):
    """Function to create model using the combined achitecture of pre-trained CNN as
    feature extractor and ViT as classifier. Default feature extractor is EfficientNetV2S
    with layers frozen."""

    base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2S(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )

    base_model.trainable = False

    TRANSFORMER_UNITS = [projection_dim * 2, projection_dim]

    resize = tf.keras.layers.Reshape((input_shape[0], input_shape[1]))
    inputs = layers.Input(shape=input_shape)(resize)
    feature_extractor = base_model(inputs)
    shape_to_encode = feature_extractor.shape[-1]
    patches_num = feature_extractor.shape[1] ** 2
    reshaped_features = layers.Reshape((patches_num, shape_to_encode))(
        feature_extractor
    )
    encoded_patches = PatchEncoder(patches_num, projection_dim)(reshaped_features)
    for _ in range(transformer_layers):
        x_1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x_1, x_1)
        # Skip connection 1.
        x_2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x_3 = layers.LayerNormalization(epsilon=1e-6)(x_2)
        # MLP.
        x_3 = mlp(x_3, hidden_units=TRANSFORMER_UNITS, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x_3, x_2])
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    output = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    output = layers.Dense(num_classes, activation="softmax")(output)
    model = tf.keras.Model(inputs=inputs, outputs=output)

    return model

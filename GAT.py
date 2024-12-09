## with visuals
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt

import os

class PositionalEncoding(tf.keras.layers.Layer):
    def call(self, inputs):
        # Ensure inputs have the required dimensions
        if len(inputs.shape) < 3:
            raise ValueError(f"Expected inputs to have at least 3 dimensions (batch_size, seq_length, feature_dim), "
                             f"but got shape {inputs.shape}.")

        seq_length = tf.shape(inputs)[1]
        feature_dim = tf.shape(inputs)[2]

        # Generate position indices and angles
        positions = tf.range(seq_length, dtype=tf.float32)[:, tf.newaxis]
        angles = 1.0 / tf.pow(10000.0, (2 * (tf.range(feature_dim // 2, dtype=tf.float32)) / tf.cast(feature_dim, tf.float32)))
        angles = tf.expand_dims(angles, axis=0)

        # Compute positional encodings
        pos_encoding = tf.concat([tf.sin(positions * angles), tf.cos(positions * angles)], axis=-1)

        batch_size = tf.shape(inputs)[0]
        pos_encoding = pos_encoding[:seq_length, :feature_dim]  # Truncate or pad to match input dimensions
        pos_encoding = tf.tile(pos_encoding[tf.newaxis, :, :], [batch_size, 1, 1])

        return inputs + pos_encoding  # Add positional encodings to the input

    def compute_output_shape(self, input_shape):
        return input_shape



def transformer_encoder(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    patches = layers.Conv2D(16, (4, 4), strides=(4, 4), activation="relu", kernel_initializer='he_normal')(inputs)
    reshaped = layers.Reshape((-1, 16))(patches)  # Reshape to (batch_size, seq_length, feature_dim)
    transformer_with_position = PositionalEncoding()(reshaped)
    transformer = layers.MultiHeadAttention(num_heads=2, key_dim=8)(transformer_with_position, transformer_with_position)
    transformer = layers.Add()([reshaped, transformer])
    output_shape = (input_shape[0] // 4, input_shape[1] // 4, 16)
    transformer_output = layers.Reshape(output_shape)(transformer)
    return tf.keras.Model(inputs, transformer_output, name="TransformerEncoder")



# Auxiliary Autoencoder
def auxiliary_autoencoder(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = layers.MaxPooling2D()(x)
    bottleneck = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = layers.UpSampling2D()(bottleneck)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = layers.UpSampling2D()(x)
    outputs = layers.Conv2D(1, (3, 3), activation='tanh', padding='same')(x)
    return tf.keras.Model(inputs, bottleneck, name="AuxiliaryAutoencoder")


# GAN Generator
def gan_generator(input_shape, transformer_output_shape, autoencoder_output_shape, target_shape=(64, 64)):
    grayscale_input = layers.Input(shape=input_shape)
    transformer_input = layers.Input(shape=transformer_output_shape)
    autoencoder_input = layers.Input(shape=autoencoder_output_shape)

    # Resize grayscale input to match target shape (64, 64) to match transformer and autoencoder outputs
    resize_grayscale = layers.Lambda(lambda x: tf.image.resize(x, target_shape),
                                     output_shape=(target_shape[0], target_shape[1], input_shape[-1]))(grayscale_input)

    # Apply Conv2D to increase depth of transformer and autoencoder inputs
    transformer_depth_increased = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(transformer_input)
    autoencoder_depth_increased = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(autoencoder_input)

    # Resize transformer and autoencoder outputs to match target shape (64, 64) for concatenation
    resize_layer = layers.Lambda(lambda x: tf.image.resize(x, target_shape),
                                 output_shape=(target_shape[0], target_shape[1], 64))

    transformer_resized = resize_layer(transformer_depth_increased)
    autoencoder_resized = resize_layer(autoencoder_depth_increased)

    # Concatenate grayscale input (now resized) with resized transformer and autoencoder inputs
    concatenated = layers.Concatenate()([resize_grayscale, transformer_resized, autoencoder_resized])

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(concatenated)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)

    rgb_output = layers.Conv2D(3, (3, 3), activation='tanh', padding='same')(x)

    return tf.keras.Model([grayscale_input, transformer_input, autoencoder_input], rgb_output, name="GANGenerator")


# GAN Discriminator
def gan_discriminator(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu')(inputs)
    x = layers.Conv2D(128, (3, 3), strides=(2, 2), activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs, x, name="GANDiscriminator")

def save_model_with_custom_layers(model, filepath):
    model.save(filepath, save_format="h5")

def load_model_with_custom_layers(filepath):
    return tf.keras.models.load_model(filepath, custom_objects={
        "PositionalEncoding": PositionalEncoding,
        "tf": tf
    })


if __name__ == "__main__":
    input_shape = (128, 128, 1)
    model = transformer_encoder(input_shape)
    save_model_with_custom_layers(model, "transformer_encoder.h5")
    loaded_model = load_model_with_custom_layers("transformer_encoder.h5")
    print(loaded_model.summary())


# Pre-trained VGG16 for Perceptual Loss
vgg = VGG16(include_top=False, weights="imagenet", input_shape=(64, 64, 3))
for layer in vgg.layers:
    layer.trainable = False


# Loss Functions
def perceptual_loss(y_true, y_pred):
    y_true_vgg = vgg(y_true)
    y_pred_vgg = vgg(y_pred)
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    perceptual_loss = tf.reduce_mean(tf.abs(y_true_vgg - y_pred_vgg))
    return mse_loss + 0.25 * perceptual_loss


def discriminator_loss(y_true, y_pred):
    return tf.keras.losses.BinaryCrossentropy(from_logits=False)(y_true, y_pred)


def generator_loss(y_pred):
    return tf.keras.losses.BinaryCrossentropy(from_logits=False)(tf.ones_like(y_pred), y_pred)


def psnr_metric(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)


# Initialize Models
IMG_HEIGHT, IMG_WIDTH = 64, 64
input_shape = (IMG_HEIGHT, IMG_WIDTH, 1)

transformer_encoder_model = transformer_encoder(input_shape)
autoencoder_model = auxiliary_autoencoder(input_shape)
transformer_output_shape = transformer_encoder_model.output_shape[1:]
autoencoder_output_shape = autoencoder_model.output_shape[1:]

gan_generator_model = gan_generator(input_shape, transformer_output_shape, autoencoder_output_shape)
gan_discriminator_model = gan_discriminator((IMG_HEIGHT, IMG_WIDTH, 3))

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=2e-4, decay_steps=1000, decay_rate=0.96
)
optimizer_g = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
optimizer_d = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

SAVE_DIR = "./saved_models"
try:
    os.makedirs(SAVE_DIR, exist_ok=True)
except OSError as e:
    print(f"Error creating save directory {SAVE_DIR}: {e}")

def compile_and_save_model(model, optimizer, loss_function, save_path):
    model.compile(optimizer=optimizer, loss=loss_function)
    model.save(save_path)

def show_generated_image(image_tensor, epoch, step):
    image = (image_tensor[0].numpy() * 0.5 + 0.5)  # Assuming images are normalized in the range [-1, 1]
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Epoch {epoch + 1}, Step {step}")
    plt.show()

def train_model(dataset, epochs=5):
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        for step, (grayscale, rgb) in enumerate(dataset):
            if grayscale.shape[-1] != 1:
                grayscale = tf.expand_dims(grayscale, axis=-1)
            transformer_features = transformer_encoder_model(grayscale)
            autoencoder_features = autoencoder_model(grayscale)

            # Train discriminator
            with tf.GradientTape() as tape_d:
                fake_images = gan_generator_model([grayscale, transformer_features, autoencoder_features])
                real_output = gan_discriminator_model(rgb)
                fake_output = gan_discriminator_model(fake_images)
                d_loss_real = discriminator_loss(tf.ones_like(real_output), real_output)
                d_loss_fake = discriminator_loss(tf.zeros_like(fake_output), fake_output)
                d_loss = d_loss_real + d_loss_fake

            grads_d = tape_d.gradient(d_loss, gan_discriminator_model.trainable_variables)
            optimizer_d.apply_gradients(zip(grads_d, gan_discriminator_model.trainable_variables))

            # Train generator
            with tf.GradientTape() as tape_g:
                fake_images = gan_generator_model([grayscale, transformer_features, autoencoder_features])
                fake_output = gan_discriminator_model(fake_images)
                g_loss = generator_loss(fake_output) + perceptual_loss(rgb, fake_images)
                psnr = psnr_metric(rgb, fake_images)  # Calculate PSNR

            grads_g = tape_g.gradient(g_loss, gan_generator_model.trainable_variables)
            optimizer_g.apply_gradients(zip(grads_g, gan_generator_model.trainable_variables))

            #generated image every 10 steps
            if step % 10 == 0:
                print(f"Step {step}: D Loss = {d_loss:.4f}, G Loss = {g_loss:.4f}, PSNR = {tf.reduce_mean(psnr):.2f}")
                show_generated_image(fake_images, epoch, step)

        compile_and_save_model(gan_generator_model, optimizer_g, generator_loss, os.path.join(SAVE_DIR, f"gan_generator_epoch_{epoch + 1}.h5"))
        compile_and_save_model(gan_discriminator_model, optimizer_d, discriminator_loss, os.path.join(SAVE_DIR, f"gan_discriminator_epoch_{epoch + 1}.h5"))
        print(f"Models saved for epoch {epoch + 1}.")

# Call the training function
train_model(dataset, epochs=5)
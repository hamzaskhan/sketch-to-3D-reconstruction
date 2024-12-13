import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG16
import numpy as np
import matplotlib.pyplot as plt

#This is the improved CNN inspired by TCP handhsake, which replaces the regular CNNs. As seen in ablation study of the paper.
class TCPCNN(layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding="same", activation="relu", **kwargs):
        super(TCPCNN, self).__init__(**kwargs)
        self.conv = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, activation=activation)
        self.global_pool = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(filters // 4, activation="relu")
        self.dense2 = layers.Dense(filters, activation="sigmoid")

    def call(self, inputs):
        conv_output = self.conv(inputs)
        channel_attention = self.global_pool(conv_output)
        channel_attention = self.dense1(channel_attention)
        channel_attention = self.dense2(channel_attention)
        channel_attention = tf.expand_dims(tf.expand_dims(channel_attention, axis=1), axis=1)
        prioritized_output = conv_output * channel_attention
        return prioritized_output


class PositionalEncoding(layers.Layer):
    def call(self, inputs):
        seq_length = tf.shape(inputs)[1]
        feature_dim = tf.shape(inputs)[2]
        positions = tf.range(seq_length, dtype=tf.float32)[:, tf.newaxis]
        div_terms = tf.pow(10000.0, (2 * tf.range(feature_dim // 2, dtype=tf.float32) / tf.cast(feature_dim, tf.float32)))
        angles = positions / div_terms
        pos_encoding = tf.concat([tf.sin(angles), tf.cos(angles)], axis=-1)
        return inputs + tf.expand_dims(pos_encoding, axis=0)


def transformer_encoder(input_shape):
    inputs = layers.Input(shape=input_shape)
    patches = layers.Conv2D(16, (4, 4), strides=(4, 4), activation="relu", kernel_initializer="he_normal")(inputs)
    reshaped = layers.Reshape((-1, 16))(patches)
    encoded = PositionalEncoding()(reshaped)
    attention_output = layers.MultiHeadAttention(num_heads=2, key_dim=8)(encoded, encoded)
    residual = layers.Add()([reshaped, attention_output])
    output_shape = (input_shape[0] // 4, input_shape[1] // 4, 16)
    reshaped_back = layers.Reshape(output_shape)(residual)
    return Model(inputs, reshaped_back, name="TransformerEncoder")


def auxiliary_autoencoder(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = TCPCNN(16, (3, 3))(inputs)
    x = layers.MaxPooling2D()(x)
    x = TCPCNN(32, (3, 3))(x)
    x = layers.MaxPooling2D()(x)
    bottleneck = TCPCNN(64, (3, 3))(x)
    x = layers.UpSampling2D()(bottleneck)
    x = TCPCNN(32, (3, 3))(x)
    x = layers.UpSampling2D()(x)
    outputs = layers.Conv2D(1, (3, 3), activation="tanh", padding="same")(x)
    return Model(inputs, bottleneck, name="AuxiliaryAutoencoder")


def gan_generator(input_shape, transformer_output_shape, autoencoder_output_shape, target_shape=(64, 64)):
    grayscale_input = layers.Input(shape=input_shape)
    transformer_input = layers.Input(shape=transformer_output_shape)
    autoencoder_input = layers.Input(shape=autoencoder_output_shape)

    resize = layers.Lambda(lambda x: tf.image.resize(x, target_shape))
    resized_grayscale = resize(grayscale_input)
    transformer_resized = resize(transformer_input)
    autoencoder_resized = resize(autoencoder_input)

    concatenated = layers.Concatenate()([resized_grayscale, transformer_resized, autoencoder_resized])
    x = TCPCNN(64, (3, 3))(concatenated)
    x = TCPCNN(32, (3, 3))(x)
    outputs = layers.Conv2D(3, (3, 3), activation="tanh", padding="same")(x)
    return Model([grayscale_input, transformer_input, autoencoder_input], outputs, name="GANGenerator")


def gan_discriminator(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = TCPCNN(64, (3, 3), strides=(2, 2))(inputs)
    x = TCPCNN(128, (3, 3), strides=(2, 2))(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return Model(inputs, outputs, name="GANDiscriminator")

# Metrics Calculation
@tf.function
def calculate_metrics(y_true, y_pred):
    psnr = tf.image.psnr(y_true, y_pred, max_val=1.0)
    ssim = tf.image.ssim(y_true, y_pred, max_val=1.0)
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    return tf.reduce_mean(psnr), tf.reduce_mean(ssim), mae

# Load Pre-trained VGG for Perceptual Loss
vgg = VGG16(include_top=False, weights="imagenet", input_shape=(64, 64, 3))
vgg.trainable = False

# Model Initialization
IMG_HEIGHT, IMG_WIDTH = 64, 64
input_shape = (IMG_HEIGHT, IMG_WIDTH, 1)

transformer = transformer_encoder(input_shape)
autoencoder = auxiliary_autoencoder(input_shape)
generator = gan_generator(input_shape, transformer.output_shape[1:], autoencoder.output_shape[1:])
discriminator = gan_discriminator((IMG_HEIGHT, IMG_WIDTH, 3))

# Optimizers with Learning Rate Schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-4, decay_steps=1000, decay_rate=0.95)
optimizer_g = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
optimizer_d = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

@tf.function
@tf.function
def train_step(grayscale, rgb, transformer, autoencoder, generator, discriminator, optimizer_g, optimizer_d, vgg):
    with tf.GradientTape(persistent=True) as tape:
        transformer_output = transformer(grayscale, training=True)
        autoencoder_output = autoencoder(grayscale, training=True)
        generated_rgb = generator([grayscale, transformer_output, autoencoder_output], training=True)

        real_output = discriminator(rgb, training=True)
        fake_output = discriminator(generated_rgb, training=True)

        real_features = vgg(rgb)
        fake_features = vgg(generated_rgb)

        perceptual_loss = tf.reduce_mean(tf.abs(real_features - fake_features))
        color_loss = tf.reduce_mean(tf.abs(rgb[:, :, :, 1:] - generated_rgb[:, :, :, 1:]))
        adv_loss_g = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(fake_output), fake_output))
        g_loss = adv_loss_g + 0.5 * perceptual_loss + 0.5 * color_loss

        real_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(real_output), real_output))
        fake_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_output), fake_output))
        d_loss = (real_loss + fake_loss) / 2

        psnr = tf.image.psnr(rgb, generated_rgb, max_val=1.0)
        ssim = tf.image.ssim(rgb, generated_rgb, max_val=1.0)
        mae = tf.reduce_mean(tf.abs(rgb - generated_rgb))

    gradients_g = tape.gradient(g_loss, generator.trainable_variables)
    optimizer_g.apply_gradients(zip(gradients_g, generator.trainable_variables))

    gradients_d = tape.gradient(d_loss, discriminator.trainable_variables)
    optimizer_d.apply_gradients(zip(gradients_d, discriminator.trainable_variables))

    del tape
    return d_loss, g_loss, tf.reduce_mean(psnr), tf.reduce_mean(ssim), mae, generated_rgb


def train_model(transformer, autoencoder, generator, discriminator, optimizer_g, optimizer_d, vgg, dataset, epochs=1, steps_per_epoch=None):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        metrics = {'d_loss': [], 'g_loss': [], 'psnr': [], 'ssim': [], 'mae': []}
        for step, (grayscale, rgb) in enumerate(dataset):
            d_loss, g_loss, psnr, ssim, mae, generated_rgb = train_step(
                grayscale, rgb, transformer, autoencoder, generator, discriminator, optimizer_g, optimizer_d, vgg
            )
            metrics['d_loss'].append(d_loss.numpy())
            metrics['g_loss'].append(g_loss.numpy())
            metrics['psnr'].append(psnr.numpy())
            metrics['ssim'].append(ssim.numpy())
            metrics['mae'].append(mae.numpy())

            if step % 10 == 0:
                print(f"Step {step}: D Loss = {d_loss:.4f}, G Loss = {g_loss:.4f}, PSNR = {psnr:.2f}, SSIM = {ssim:.2f}, MAE = {mae:.4f}")

            if steps_per_epoch and step >= steps_per_epoch - 1:
                break

        avg_d_loss = np.mean(metrics['d_loss'])
        avg_g_loss = np.mean(metrics['g_loss'])
        avg_psnr = np.mean(metrics['psnr'])
        avg_ssim = np.mean(metrics['ssim'])
        avg_mae = np.mean(metrics['mae'])

        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Avg D Loss = {avg_d_loss:.4f}")
        print(f"  Avg G Loss = {avg_g_loss:.4f}")
        print(f"  Avg PSNR = {avg_psnr:.2f}")
        print(f"  Avg SSIM = {avg_ssim:.2f}")
        print(f"  Avg MAE = {avg_mae:.4f}")

        # Visualize RGB, Grayscale, and Generated images
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(rgb[0].numpy())
        axes[0].set_title("RGB")
        axes[1].imshow(grayscale[0].numpy().squeeze(), cmap="gray")
        axes[1].set_title("Grayscale")
        axes[2].imshow((generated_rgb[0].numpy() * 0.5 + 0.5))  # Denormalize for visualization
        axes[2].set_title("Generated")
        plt.show()

dataset = load_paired_dataset(grayscale_dir, rgb_dir, batch_size=BATCH_SIZE)
train_model(
    transformer=transformer,
    autoencoder=autoencoder,
    generator=generator,
    discriminator=discriminator,
    optimizer_g=optimizer_g,
    optimizer_d=optimizer_d,
    vgg=vgg,
    dataset=dataset,
    epochs=70,
    steps_per_epoch=94
)

import tensorflow as tf

def make_generator(latent_size) :
    g_init = tf.keras.initializers.TruncatedNormal(stddev = 0.02)
    hidden_space = tf.keras.Input(shape = (latent_size, ), name = "hidden_space")

    outputs = tf.keras.layers.Dense(7 * 7 * 1024, kernel_initializer = g_init)(hidden_space)
    outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = tf.keras.layers.Reshape(target_shape = (7, 7, 1024, ))(outputs)

    outputs = tf.keras.layers.Conv2DTranspose(128, 4, 2, "same", kernel_initializer = g_init)(outputs) # (14, 14, 128, )
    outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = tf.keras.activations.relu(outputs)

    outputs = tf.keras.layers.Conv2DTranspose(64, 4, 1, "same", kernel_initializer = g_init)(outputs) # (14, 14, 64, )
    outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = tf.keras.activations.relu(outputs)

    outputs = tf.keras.layers.Conv2DTranspose(32, 4, 2, "same", kernel_initializer = g_init)(outputs) # (28, 28, 32, )
    outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = tf.keras.activations.relu(outputs)

    outputs = tf.keras.layers.Conv2DTranspose(1, 4, 1, "same", kernel_initializer = g_init)(outputs) # (28, 28, 1, )

    images = tf.keras.layers.Activation(tf.keras.activations.tanh, name = "images")(outputs)

    return tf.keras.Model(inputs = [hidden_space], outputs = [images], name = "Generator")

def make_discriminator(latent_size) :
    d_init = tf.keras.initializers.TruncatedNormal(stddev = 0.002)
    
    # Encoder
    images = tf.keras.Input(shape = (28, 28, 1, ), name = "images")

    outputs = tf.keras.layers.Conv2D(32, 4, 2, "same", kernel_initializer = d_init)(images) # (14, 14, 32, )
    outputs = tf.keras.activations.relu(outputs)

    outputs = tf.keras.layers.Conv2D(64, 4, 1, "same", kernel_initializer = d_init)(outputs) # (14, 14, 64, )
    outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = tf.keras.activations.relu(outputs)

    outputs = tf.keras.layers.Conv2D(128, 4, 2, "same", kernel_initializer = d_init)(outputs) # (7, 7, 128, )
    outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = tf.keras.activations.relu(outputs)

    outputs = tf.keras.layers.Flatten()(outputs)
    hidden_space = tf.keras.layers.Dense(latent_size, kernel_initializer = d_init, name = "hidden_space")(outputs) # (latent_size, )

    outputs = tf.keras.layers.Dense(1024 * 7 * 7, kernel_initializer = d_init)(hidden_space) 
    outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = tf.keras.activations.relu(outputs)

    outputs = tf.keras.layers.Reshape(target_shape = (7, 7, 1024, ))(outputs) # (7, 7, 64, )

    outputs = tf.keras.layers.Conv2DTranspose(128, 4, 2, "same", kernel_initializer = d_init)(outputs) # (14, 14, 128, )
    outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = tf.keras.activations.relu(outputs)

    outputs = tf.keras.layers.Conv2DTranspose(64, 4, 1, "same", kernel_initializer = d_init)(outputs) # (14, 14, 64, )
    outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = tf.keras.activations.relu(outputs)

    outputs = tf.keras.layers.Conv2DTranspose(32, 4, 2, "same", kernel_initializer = d_init)(outputs) # (28, 28, 32, )
    outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = tf.keras.activations.relu(outputs)

    outputs = tf.keras.layers.Conv2DTranspose(1, 4, 1, "same", kernel_initializer = d_init)(outputs) # (28, 28, 1, )

    reconstructed_images = tf.keras.layers.Activation(tf.keras.activations.tanh, 
        name = "reconstructed_images")(outputs)

    return tf.keras.Model(inputs = [images], outputs = [reconstructed_images, hidden_space], 
        name = "Discriminator")
import os
from math import ceil

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.keras.datasets import mnist
import tensorflow.keras.backend as K

from Checkpoint import load_checkpoint
from Utils import make_noise, korea_time, save_summary, make_folders_for_model, save_initial_model_info

class LSGAN() :
    # https://arxiv.org/pdf/1611.04076.pdf
    def __init__(self, **kwargs) :
        self.batch_size = kwargs["batch_size"]
        self.latent_size = kwargs["latent_size"]
        self.lr = kwargs["lr"]
        assert isinstance(self.lr, float) or (isinstance(self.lr, list) and len(self.lr) == 2)

        if isinstance(self.lr, list) :
            self.g_lr = self.lr[0]
            self.d_lr = self.lr[1]
        else :
            self.g_lr = self.d_lr = self.lr

        self.smoothing = max(kwargs["smoothing"], 1.0)
        self.initial_epoch = 1

    def train(self, **kwargs) :
        interval = kwargs["interval"]
        model_ckpt_path, model_images_path, model_logs_path, model_result_file = make_folders_for_model(kwargs['folder'])

        self.generator = self.make_generator(**kwargs)
        self.discriminator = self.make_discriminator(**kwargs)
        
        train_dataset = self.get_dataset()
        num_batches = ceil(self.num_train / self.batch_size)

        d_epoch_loss = []
        g_epoch_loss = []

        training_progbar = tf.keras.utils.Progbar(target = self.num_train)

        save_initial_model_info({'generator' : self.generator, 'discriminator' : self.discriminator}, 
            model_logs_path, model_ckpt_path, **kwargs)

        count = 0
    
        self.g_opt = tf.keras.optimizers.Adam(lr = self.g_lr, beta_1 = kwargs["beta_1"])
        self.d_opt = tf.keras.optimizers.Adam(lr = self.d_lr, beta_1 = kwargs["beta_1"])
        self.loss_function = tf.keras.losses.MeanSquaredError()

        ckpt = tf.train.Checkpoint(
            g_opt = self.g_opt, d_opt = self.d_opt, g_model = self.generator, d_model = self.discriminator)

        if kwargs["ckpt_path"] is not None :
            fname, self.initial_epoch = load_checkpoint(**kwargs)
            print("\nCheckpoint File : {}\n".format(fname))
            ckpt.mapped = {
                "g_opt" : self.g_opt, "d_opt" : self.d_opt, "g_model" : self.generator, "d_model" : self.discriminator}
            ckpt.restore(fname)

            self.g_lr = self.g_opt.get_config()["learning_rate"]
            self.d_lr = self.d_opt.get_config()["learning_rate"]

        for epoch in range(self.initial_epoch, self.initial_epoch + kwargs["epochs"]) :
            count += 1

            start_time = korea_time()
            for real_images in train_dataset :
                num_images = K.int_shape(real_images)[0]
                g_loss = (self.train_G(num_images)).numpy()
                d_loss = (self.train_D(real_images)).numpy()
                
                d_epoch_loss.append(d_loss)
                g_epoch_loss.append(g_loss)

                training_progbar.add(num_images)

            end_time = korea_time()
            training_progbar.update(0) # Progress bar 초기화

            d_mean_loss = np.mean(d_epoch_loss, axis = 0)
            g_mean_loss = np.mean(g_epoch_loss, axis = 0)

            ckpt_prefix = os.path.join(model_ckpt_path, "Epoch-{}_G-Loss-{:.6f}_D-Loss-{:.6f}".format(epoch, g_mean_loss, d_mean_loss))
            ckpt.save(file_prefix = ckpt_prefix)

            print("Epoch = [{:5d}]\tGenerator Loss = [{:8.6f}]\tDiscriminator Loss = [{:8.6f}]\n".format(
                epoch, g_mean_loss, d_mean_loss))

            # model result 저장
            str_ = "Epoch = [{:5d}] - End Time [ {} ]\n".format(
                epoch, str(end_time.strftime("%Y / %m / %d   %H:%M:%S")))
            str_ += "Elapsed Time = {}\n".format(end_time - start_time)
            str_ += "Generator Learning Rate = [{:.6f}] - Discriminator Learning Rate = [{:.6f}]\n".format(
                self.g_lr, self.d_lr)
            str_ += "Generator Loss : [{:8.6f}] - Discriminator Loss : [{:8.6f}] - Sum : [{:8.6f}]\n".format(
                g_mean_loss, d_mean_loss, g_mean_loss + d_mean_loss)
            str_ += " - " * 15 + "\n\n"
            
            with open(model_result_file, "a+", encoding = 'utf-8') as fp :
                fp.write(str_)

            if count == interval :
                fname = os.path.join(model_images_path, "{}.png".format(epoch))
                self.plot_images(fname)
                
                count = 0

            d_epoch_loss = []
            g_epoch_loss = []

    @tf.function
    def train_G(self, num_images) :
        noise_sample = make_noise(num_images, self.latent_size)

        with tf.GradientTape() as gen_tape :
            fake_images = self.generator(noise_sample, training = True)
            fake_logits = self.discriminator(fake_images, training = True)

            g_loss = self.get_generator_loss(fake_logits)

        # Gradient 계산
        gradients = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(gradients, self.generator.trainable_variables))

        return g_loss

    @tf.function
    def train_D(self, real_images) :
        num_images = K.int_shape(real_images)[0]
        noise_sample = make_noise(num_images, self.latent_size)

        with tf.GradientTape() as disc_tape :
            fake_images = self.generator(noise_sample, training = True)

            real_logits = self.discriminator(real_images, training = True)
            fake_logits = self.discriminator(fake_images, training = True)

            d_loss = self.get_discriminator_loss(real_logits, fake_logits)

        disc_gradients = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

        return d_loss

    def get_discriminator_loss(self, real_logits, fake_logits) :
        real_loss = self.loss_function(tf.ones_like(real_logits) * self.smoothing, real_logits) * 0.5
        fake_loss = self.loss_function(tf.zeros_like(fake_logits), fake_logits) * 0.5
        return real_loss + fake_loss

    def get_generator_loss(self, fake_logits) :
        return self.loss_function(tf.ones_like(fake_logits) * self.smoothing, fake_logits) * 0.5
        
    def get_dataset(self) :
        (train_X, train_y), (test_X, test_y) = mnist.load_data() # (28, 28, 1)
        train_X = np.concatenate([train_X, test_X], axis = 0)
        train_X = train_X.reshape(train_X.shape[0], 28, 28, 1).astype('float32')

        # 0 ~ 255 --- (X - 127.5) ---> -127.5 ~ 127.5 --- (X / 127.5) ---> -1 ~ 1
        train_X = (train_X - 127.5) / 127.5

        self.num_train = K.int_shape(train_X)[0]

        return tf.data.Dataset.from_tensor_slices(train_X).shuffle(self.num_train).batch(self.batch_size)

    def plot_images(self, fname) :
        fig = plt.figure(figsize = (8, 8))
        noise_sample = make_noise(100, self.latent_size)
        fake_images = self.generator(noise_sample, training = False)

        for i in range(fake_images.shape[0]) :
            plt.subplot(10, 10, i + 1)
            plt.imshow(fake_images[i, :, :, 0] * 127.5 + 127.5, cmap = 'gray')
            plt.axis('off')

        plt.savefig(fname)
        plt.close()

    def make_generator(self, **kwargs) :
        z = tf.keras.Input(shape = (kwargs['latent_size'], ), name = 'z')

        outputs = tf.keras.layers.Dense(units = 7 * 7 * 256, use_bias = kwargs["use_bias"], 
            name = 'input_prog')(z) 
        outputs = tf.keras.layers.BatchNormalization()(outputs)
        outputs = tf.keras.activations.relu(outputs)
        
        outputs = tf.keras.layers.Reshape(target_shape = (7, 7, 256, ))(outputs) # (7, 7, 256)

        outputs = tf.keras.layers.Conv2DTranspose(128, 5, 1, "same", use_bias = kwargs["use_bias"])(outputs) # (7, 7, 128)
        outputs = tf.keras.layers.BatchNormalization()(outputs)
        outputs= tf.keras.activations.relu(outputs)

        outputs = tf.keras.layers.Conv2DTranspose(64, 5, 2, "same", use_bias = kwargs["use_bias"])(outputs) # (14, 14, 64)
        outputs = tf.keras.layers.BatchNormalization()(outputs)
        outputs = tf.keras.activations.relu(outputs)

        fake_images = tf.keras.layers.Conv2DTranspose(1, 5, 2, "same", activation = "tanh", 
            use_bias = kwargs["use_bias"])(outputs) # (28, 28, 1)
        assert K.int_shape(fake_images) == (None, 28, 28, 1)

        return tf.keras.Model(inputs = [z], outputs = [fake_images], name = 'Generator')

    def make_discriminator(self, **kwargs) :
        images = tf.keras.Input(shape = (28, 28, 1, ), name = "images")

        outputs = tf.keras.layers.Conv2D(64, 5, 2, "same")(images)
        outputs = tf.keras.activations.relu(outputs, alpha = kwargs["slope"])
        if kwargs["dropout"] != 0 :
            outputs = tf.keras.layers.Dropout(kwargs["dropout"])(outputs)

        outputs = tf.keras.layers.Conv2D(128, 5, 2, "same")(outputs)
        outputs = tf.keras.layers.BatchNormalization()(outputs)
        outputs = tf.keras.activations.relu(outputs, alpha = kwargs["slope"])
        if kwargs["dropout"] != 0 :
            outputs = tf.keras.layers.Dropout(kwargs["dropout"])(outputs)

        outputs = tf.keras.layers.Conv2D(256, 5, 2, "same")(outputs)
        outputs = tf.keras.layers.BatchNormalization()(outputs)
        outputs = tf.keras.activations.relu(outputs, alpha = kwargs["slope"])
        if kwargs["dropout"] != 0 :
            outputs = tf.keras.layers.Dropout(kwargs["dropout"])(outputs)

        outputs = tf.keras.layers.Flatten()(outputs)
        outputs = tf.keras.layers.Dense(units = 512)(outputs)
        outputs = tf.keras.layers.BatchNormalization()(outputs)
        outputs = tf.keras.activations.relu(outputs, alpha = kwargs["slope"])

        logits_y = tf.keras.layers.Dense(units = 1, name = 'logits_y')(outputs)
        
        return tf.keras.Model(inputs = [images], outputs = [logits_y], name = 'Discriminator')

    def test(self, *args, **kwargs) :
        self.generator = self.make_generator(**kwargs)

        ckpt = tf.train.Checkpoint(g_model = self.generator)
        fname, _ = load_checkpoint(**kwargs)
        print("\nCheckpoint File : {}\n".format(fname))

        # model만 불러옴
        ckpt.mapped = {"g_model" : self.generator}
        ckpt.restore(fname).expect_partial()

        _, model_images_path, _, _ = make_folders_for_model(kwargs['folder'])
        fname = os.path.join(model_images_path, "Test.png")

        self.plot_images(fname)
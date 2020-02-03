import os
from math import ceil

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.keras.datasets import mnist
import tensorflow.keras.backend as K

from Checkpoint import load_checkpoint
from Utils import make_noise, korea_time, save_summary, make_folders_for_model, save_initial_model_info

class WGAN_GP() :
    # https://arxiv.org/pdf/1701.07875.pdf (WGAN)
    # https://arxiv.org/pdf/1704.00028.pdf (WGAN-GP)
    def __init__(self) :
        self.batch_size = 64
        self.n_critic = 5
        self.latent_size = 100
        self.gp_coefficient = 10
        self.g_lr = self.c_lr = 0.0001
        self.initial_epoch = 1

    def train(self, **kwargs) :
        interval = kwargs["interval"]
        model_ckpt_path, model_images_path, model_logs_path, model_result_file = make_folders_for_model(kwargs['folder'])

        self.generator = self.make_generator()
        self.critic = self.make_critic()

        train_dataset = self.get_dataset()
        num_batches = ceil(self.num_train / self.batch_size)

        c_epoch_loss = []
        g_epoch_loss = []

        training_progbar = tf.keras.utils.Progbar(target = self.num_train)

        save_initial_model_info({'generator' : self.generator, 'critic' : self.critic}, 
            model_logs_path, model_ckpt_path, **kwargs)

        count = 0
    
        self.g_opt = tf.keras.optimizers.Adam(lr = self.g_lr, beta_1 = 0, beta_2 = 0.9)
        self.c_opt = tf.keras.optimizers.Adam(lr = self.c_lr, beta_1 = 0, beta_2 = 0.9)

        ckpt = tf.train.Checkpoint(
            g_opt = self.g_opt, c_opt = self.c_opt, g_model = self.generator, c_model = self.critic)

        if kwargs["ckpt_path"] is not None :
            fname, self.initial_epoch = load_checkpoint(**kwargs)
            print("\nCheckpoint File : {}\n".format(fname))
            ckpt.mapped = {
                "g_opt" : self.g_opt, "c_opt" : self.c_opt, "g_model" : self.generator, "c_model" : self.critic}
            ckpt.restore(fname)

            self.g_lr = self.g_opt.get_config()["learning_rate"]
            self.c_lr = self.c_opt.get_config()["learning_rate"]

        for epoch in range(self.initial_epoch, self.initial_epoch + 50000) :
            count += 1

            start_time = korea_time()
            num_batch = 0 # 64 * 5 = 320
            mult = self.n_critic * self.batch_size
            num_dataset = 0 # 60000
            real_images_list = []
            
            for real_images in train_dataset :
                # self.n_critic개 만큼의 image dataset을 불러옴
                real_images_list.append(real_images)
                num_images = K.int_shape(real_images)[0]
                num_batch += num_images
                num_dataset += num_images

                if (num_batch == mult) or (num_dataset == self.num_train) :
                    critic_loss_list = [(self.train_D(real_images)).numpy() for real_images in real_images_list]
                    g_loss = (self.train_G()).numpy()

                    c_epoch_loss.extend(critic_loss_list)
                    g_epoch_loss.append(g_loss)

                    training_progbar.add(num_batch)

                    if num_dataset == self.num_train :
                        break

                    num_batch = 0
                    real_images_list = []

            end_time = korea_time()
            training_progbar.update(0) # Progress bar 초기화

            c_mean_loss = np.mean(c_epoch_loss, axis = 0)
            g_mean_loss = np.mean(g_epoch_loss, axis = 0)

            ckpt_prefix = os.path.join(model_ckpt_path, "Epoch-{}_G-Loss-{:.6f}_C-Loss-{:.6f}".format(epoch, g_mean_loss, c_mean_loss))
            ckpt.save(file_prefix = ckpt_prefix)

            print("Epoch = [{:5d}]\tGenerator Loss = [{:8.6f}]\tCritic Loss = [{:8.6f}]\n".format(
                epoch, g_mean_loss, c_mean_loss))

            # model result 저장
            str_ = "Epoch = [{:5d}] - End Time [ {} ]\n".format(
                epoch, str(end_time.strftime("%Y / %m / %d   %H:%M:%S")))
            str_ += "Elapsed Time = {}\n".format(end_time - start_time)
            str_ += "Generator Learning Rate = [{:.6f}] - Critic Learning Rate = [{:.6f}]\n".format(
                self.g_lr, self.c_lr)
            str_ += "Generator Loss : [{:8.6f}] - Critic Loss : [{:8.6f}] - Sum : [{:8.6f}]\n".format(
                g_mean_loss, c_mean_loss, g_mean_loss + c_mean_loss)
            str_ += " - " * 15 + "\n\n"
            
            with open(model_result_file, "a+", encoding = 'utf-8') as fp :
                fp.write(str_)

            if count == interval :
                fname = os.path.join(model_images_path, "{}.png".format(epoch))
                self.plot_images(fname)
                
                count = 0

            c_epoch_loss = []
            g_epoch_loss = []

    @tf.function
    def train_G(self) :
        noise = make_noise(self.batch_size, self.latent_size)
        with tf.GradientTape() as gen_tape :
            fake_images = self.generator(noise, training = True)
            fake_logits = self.critic(fake_images, training = True)
            g_loss = (-1.0) * K.mean(fake_logits)

        gen_gradients = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))

        return g_loss

    @tf.function
    def train_D(self, real_images) :
        num_images = K.int_shape(real_images)[0]
        noise = make_noise(num_images, self.latent_size)

        with tf.GradientTape() as critic_tape :
            fake_images = self.generator(noise, training = True)

            alpha = tf.random.uniform((num_images, 1, 1, 1)) # [0, 1]
            other_samples = alpha * real_images + ((1 - alpha) * fake_images)

            fake_logits = self.critic(fake_images, training = True)
            real_logits = self.critic(real_images, training = True)
            other_logits = self.critic(other_samples, training = True)

            critic_loss = K.mean(fake_logits) + (-1.0) * K.mean(real_logits)
  
            # gradient penalty
            sample_gradients = K.gradients(other_logits, other_samples)
            l2_norm = K.sqrt(K.sum(K.square(sample_gradients), axis = [1, 2, 3]))
            gp = K.mean(K.square(l2_norm - 1.0))

            # Critic Loss = fake_loss - real_loss + lambda * gp
            total_loss = critic_loss + self.gp_coefficient * gp

        # Update Critic
        critic_gradients = critic_tape.gradient(total_loss, self.critic.trainable_variables)
        self.c_opt.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

        return total_loss

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

    def make_generator(self) :
        z = tf.keras.Input(shape = (self.latent_size, ), name = 'z')

        outputs = tf.keras.layers.Dense(units = 7 * 7 * 256, use_bias = False, 
            name = 'input_prog')(z) 
        outputs = tf.keras.layers.BatchNormalization()(outputs)
        outputs = tf.keras.activations.relu(outputs, alpha = 0.2)
        
        outputs = tf.keras.layers.Reshape(target_shape = (7, 7, 256, ))(outputs) # (7, 7, 256)

        outputs = tf.keras.layers.Conv2DTranspose(128, 5, 1, "same", use_bias = False)(outputs) # (7, 7, 128)
        outputs = tf.keras.layers.BatchNormalization()(outputs)
        outputs = tf.keras.activations.relu(outputs, alpha = 0.2)

        outputs = tf.keras.layers.Conv2DTranspose(64, 5, 2, "same", use_bias = False)(outputs) # (14, 14, 64)
        outputs = tf.keras.layers.BatchNormalization()(outputs)
        outputs = tf.keras.activations.relu(outputs, alpha = 0.2)

        fake_images = tf.keras.layers.Conv2DTranspose(1, 5, 2, "same", activation = "tanh", 
            use_bias = False)(outputs) # (28, 28, 1)
        assert K.int_shape(fake_images) == (None, 28, 28, 1)

        return tf.keras.Model(inputs = [z], outputs = [fake_images], name = 'Generator')

    def make_critic(self) :
        images = tf.keras.Input(shape = (28, 28, 1, ), name = "images")

        outputs = tf.keras.layers.Conv2D(64, 5, 2, "same")(images)
        outputs = tf.keras.activations.relu(outputs, alpha = 0.2)

        outputs = tf.keras.layers.Conv2D(128, 5, 2, "same")(outputs)
        outputs = tf.keras.activations.relu(outputs, alpha = 0.2)

        outputs = tf.keras.layers.Conv2D(256, 5, 2, "same")(outputs)
        outputs = tf.keras.activations.relu(outputs, alpha = 0.2)

        outputs = tf.keras.layers.Flatten()(outputs)
        outputs = tf.keras.layers.Dense(units = 512)(outputs)
        outputs = tf.keras.activations.relu(outputs, alpha = 0.2)

        logits_y = tf.keras.layers.Dense(units = 1, name = 'logits_y')(outputs)
        
        return tf.keras.Model(inputs = [images], outputs = [logits_y], name = 'critic')
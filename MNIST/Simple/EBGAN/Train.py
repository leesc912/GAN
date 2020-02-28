from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import tensorflow.keras.backend as K

from Model import make_generator, make_discriminator
from Checkpoint import load_checkpoint
from Utils import make_noise, korea_time, check_valid_path, create_folder, save_model_info

class Trainer() :
    # https://arxiv.org/pdf/1609.03126.pdf
    def __init__(self, **kwargs) :
        self.latent_size = 100
        self.initial_epoch = 1
        self.epochs = 50000
        self.batch_size = 64
        self.lr = 0.0002

        self.margin = 1
        self.pt_ratio = 0.1

        self.ckpt_path = kwargs["ckpt_path"]
        self.ckpt_epoch = kwargs["ckpt_epoch"]

        self.log_folder, self.ckpt_folder, self.image_folder = create_folder(kwargs["result_folder"])
        self.training_result_file = self.log_folder / "training_result.txt"

        self.gen = make_generator(self.latent_size)
        self.disc = make_discriminator(self.latent_size)

        # kwargs 값 저장
        msg = ""
        for k, v in list(kwargs.items()) :
            msg += "{} = {}\n".format(k, v)
        msg += "new model checkpoint path = {}\n".format(self.ckpt_folder)
        with (self.log_folder / "model_settings.txt").open("w", encoding = "utf-8") as fp :
            fp.write(msg)

        self.dataset, self.num_train = self.get_dataset(self.batch_size)

        save_model_info({"Generator" : self.gen, "Discriminator" : self.disc}, self.log_folder)

    def get_dataset(self, batch_size) :
        (train_X, _), (test_X, _) = mnist.load_data() # (28, 28, 3)
        train_X = np.concatenate([train_X, test_X], axis = 0)
        train_X = train_X.reshape(train_X.shape[0], 28, 28, 1).astype('float32')

        train_X = (train_X - 127.5) / 127.5

        num_train = train_X.shape[0]
        dataset = tf.data.Dataset.from_tensor_slices(train_X)
        dataset = dataset.cache().shuffle(num_train).batch(batch_size).prefetch(1)
        
        return dataset, num_train

    def start(self) :
        self.train()

    def train(self) :
        self.g_opt = tf.keras.optimizers.Adam(lr = self.lr, beta_1 = 0.5)
        self.d_opt = tf.keras.optimizers.Adam(lr = self.lr, beta_1 = 0.5)
        self.g_loss_metric = tf.keras.metrics.Mean(name = "g_loss")
        self.d_loss_metric = tf.keras.metrics.Mean(name = "d_loss")

        ckpt = tf.train.Checkpoint(generator = self.gen, discriminator = self.disc, genenerator_optimizer = self.g_opt, 
            discriminator_optimizer = self.d_opt)
        
        if self.ckpt_path is not None :
            fname, self.initial_epoch = load_checkpoint(Path(self.ckpt_path).resolve(), self.ckpt_epoch)
            print("\nCheckpoint File : {}\n".format(fname))
            ckpt.mapped = {"generator" : self.gen, "discriminator" : self.disc, "generator_optimizer" : self.g_opt,
                "discriminator_optimizer" : self.d_opt}
            ckpt.restore(fname)

            self.lr = self.g_opt.get_config()["learning_rate"]

        progbar = tf.keras.utils.Progbar(target = self.num_train)
        for epoch in range(self.initial_epoch, self.initial_epoch + self.epochs) :
            self.g_loss_metric.reset_states()
            self.d_loss_metric.reset_states()

            start_time = korea_time(None)
            for images in self.dataset :
                num_images = K.int_shape(images)[0]
                self.train_D(images)
                self.train_G(num_images)
                progbar.add(num_images)

            end_time = korea_time(None)
            progbar.update(0) # Progress bar 초기화

            g_loss = self.g_loss_metric.result()
            d_loss = self.d_loss_metric.result()

            ckpt_prefix = self.ckpt_folder / "Epoch-{}_gLoss-{:.6f}_dLoss-{:.6f}".format(epoch, g_loss, d_loss)
            ckpt.save(file_prefix = ckpt_prefix)

            print("Epoch = [{:5d}]  G_loss = [{:8.6f}]  D_loss = [{:8.6f}]\n".format(epoch, g_loss, d_loss))

            # model result 저장            
            with self.training_result_file.open("a+", encoding = 'utf-8') as fp :
                str_ = "Epoch = [{:5d}] - End Time [ {} ]\n".format(
                    epoch, str(end_time.strftime("%Y / %m / %d   %H:%M:%S")))
                str_ += "Elapsed Time = {}\n".format(end_time - start_time)
                str_ += "Learning Rate = [{:.6f}]\n".format(self.lr)
                str_ += "g_loss = [{:8.6f}]   d_loss = [{:8.6f}]\n".format(g_loss, d_loss)
                str_ += " - " * 15 + "\n\n"
                fp.write(str_)

            fname = self.image_folder / "{}.png".format(epoch)
            self.plot_images(fname)

    def get_l2_norm(self, s) :
        return tf.math.sqrt(tf.reduce_sum(tf.math.square(s), 1, keepdims = True))

    def repelling_regularizer(self, s1, s2) :
        batch_size = tf.cast(tf.shape(s1)[0], tf.float32)
        s1 = s1 / self.get_l2_norm(s1)
        s2 = s2 / self.get_l2_norm(s2)

        cosine_sim = tf.linalg.matmul(s1, s2, transpose_b = True)

        # 대각선에 존재하는 원소 제외
        cosine_sim -= tf.linalg.diag(tf.linalg.diag_part(cosine_sim))
        pt_loss = tf.reduce_sum(cosine_sim) / (batch_size * (batch_size - 1))

        return pt_loss

    def mse(self, x, y) :
        return tf.math.sqrt(tf.nn.l2_loss(x - y)) / tf.cast(tf.shape(x)[0], tf.float32)

    @tf.function(input_signature = [tf.TensorSpec((), tf.int32)])
    def train_G(self, num_images) :
        samples = make_noise(num_images, self.latent_size)

        with tf.GradientTape() as gen_tape :
            fake_images = self.gen(samples, training = True)
            reconstructed_images, hidden_space = self.disc(fake_images, training = True)

            loss = self.mse(fake_images, reconstructed_images)
            pt = self.repelling_regularizer(hidden_space, hidden_space)
            loss += self.pt_ratio * pt

            self.g_loss_metric.update_state(loss)

        gen_grads = gen_tape.gradient(loss, self.gen.trainable_variables)
        self.g_opt.apply_gradients(zip(gen_grads, self.gen.trainable_variables))

    @tf.function(input_signature = [tf.TensorSpec((None, 28, 28, 1), tf.float32)])
    def train_D(self, real_images) :
        num_images = tf.shape(real_images)[0]
        samples = make_noise(num_images, self.latent_size)

        with tf.GradientTape() as disc_tape :
            fake_images = self.gen(samples, training = True)

            reconstructed_real_images, _ = self.disc(real_images, training = True)
            reconstructed_fake_images, _ = self.disc(fake_images, training = True)

            loss_real = self.mse(real_images, reconstructed_real_images)
            loss_fake = self.mse(fake_images, reconstructed_fake_images)
            loss_fake = tf.math.maximum(tf.cast(self.margin, tf.float32) - loss_fake, 0.0)

            loss = loss_real + loss_fake
            self.d_loss_metric.update_state(loss)
            
        disc_grads = disc_tape.gradient(loss, self.disc.trainable_variables)
        self.d_opt.apply_gradients(zip(disc_grads, self.disc.trainable_variables))

    def plot_images(self, fname) :
        fig = plt.figure(figsize = (8, 8))
        samples = make_noise(100, self.latent_size)
        fake_images = self.gen(samples, training = False)

        for i in range(fake_images.shape[0]) :
            plt.subplot(10, 10, i + 1)
            plt.imshow(fake_images[i, :, :, 0] * 127.5 + 127.5, cmap = "gray")
            plt.axis('off')

        plt.savefig(fname)
        plt.close()
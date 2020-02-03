import os
from datetime import timedelta, datetime
import tensorflow as tf

def korea_time() :
    return datetime.utcnow() + timedelta(hours = 9)

def make_noise(num_images, latent_size) :
    # [0, 1) 범위의 값들을 생성
    return tf.random.normal((num_images, latent_size))

def make_folder(path, return_path = True) :
    if not os.path.exists(path) :
        os.mkdir(path)

    if return_path :
        return path

def save_summary(model, fname, file_mode = 'w') :
    stringList = []
    model.summary(print_fn = lambda x : stringList.append(x))
    model_summary = "\n".join(stringList)

    with open(fname, mode = file_mode, encoding = 'utf-8') as fp :
        fp.write(model_summary)

def make_folders_for_model(root_folder) :
    make_folder(root_folder, False)

    current_time = str(korea_time().strftime('%Y-%m-%d__%H%M%S'))
    result_folder = make_folder(os.path.join(root_folder, current_time))

    model_images_path = make_folder(os.path.join(result_folder, "images"))
    model_logs_path = make_folder(os.path.join(result_folder, "logs"))
    model_result_file = os.path.join(model_logs_path, "model_result.txt")
    model_ckpt_path = make_folder(os.path.join(result_folder, "ckpt"))

    return model_ckpt_path, model_images_path, model_logs_path, model_result_file

def save_initial_model_info(model_dict, model_logs_path, model_ckpt_path, **kwargs) :
    model_info_file = os.path.join(model_logs_path, "model_info.txt")
    summary_fname = os.path.join(model_logs_path, "model_summary.txt")

    # kwargs
    with open(model_info_file, "a+", encoding = 'utf-8') as fp :
        strList = ["{} = {}".format(key, value) for key, value in kwargs.items()]
        strList.append("{} = {}".format("new model checkpoint path", model_ckpt_path))
        strList.append("\n" * 4)

        fp.write("\n".join(strList))

    for model_name, model in model_dict.items() :
        # Summary
        strList = []
        model.summary(print_fn = lambda x : strList.append(x))
        model_summary = "\n".join(strList)
            
        with open(summary_fname, "a+", encoding = 'utf-8') as fp :
            fp.write(model_summary)
            fp.write("\n" * 3)

        # Plot
        plot_fname = os.path.join(model_logs_path, "{}_images.png".format(model_name))
        tf.keras.utils.plot_model(model, to_file = plot_fname)
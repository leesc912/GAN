from pathlib import Path
from datetime import timedelta, datetime
import tensorflow as tf

def korea_time(time_format = "%Y/%m/%d %H:%M:%S") :
    kt = datetime.utcnow() + timedelta(hours = 9)
    if time_format is not None :
        return kt.strftime(time_format)
    return kt

def make_noise(num_images, latent_size) :
    return tf.random.normal((num_images, latent_size))

def check_valid_path(file_path) :
    if not file_path.exists() :
        raise FileNotFoundError("\n[ {} ] 경로가 존재하지 않습니다.\n".format(file_path))

def create_folder(folder_location) :
    root_folder = Path.cwd() / "Result" if folder_location is None else Path(folder_location).resolve()
    check_valid_path(root_folder.parent)

    if not root_folder.exists() :
        root_folder.mkdir()

    result_folder = root_folder / korea_time("%Y%m%d__%H%M%S")
    log_folder = result_folder / "logs"
    ckpt_folder = result_folder / "ckpt"
    image_folder = result_folder / "images"
        
    for folder in [result_folder, log_folder, ckpt_folder, image_folder] :
        if not folder.exists() :
            folder.mkdir()

    return log_folder, ckpt_folder, image_folder

def save_model_info(model_dict, model_logs_path) :
    summary_fname = model_logs_path / "model_summary.txt"
    for model_name, model in model_dict.items() :
        # Summary
        strList = []
        model.summary(print_fn = lambda x : strList.append(x))
        model_summary = "\n".join(strList)
            
        with summary_fname.open("a+", encoding = 'utf-8') as fp :
            fp.write(model_summary)
            fp.write("\n" * 3)

        # Plot
        plot_fname = model_logs_path / "{}.png".format(model_name)
        tf.keras.utils.plot_model(model, to_file = str(plot_fname))
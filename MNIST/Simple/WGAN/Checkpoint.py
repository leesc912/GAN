import os, glob
import re
import tensorflow as tf

def get_epoch(fname) :
    result = re.findall("epoch-\d+", fname.lower())

    if len(result) :
        return int(result[0].split('-')[1])
    else :
        print("\n{}에서 'epoch-\\d+' 형식의 epoch를 찾지 못했습니다.".format(fname.lower()))
        return 0

def find_checkpoint_file(ckpt_path, epoch) :
    # file name을 모두 소문자로 변환
    ckpt_files = [fname for fname in glob.glob(os.path.join(ckpt_path, "*"))]
    ckpt_files.sort(reverse = True)

    ckpt = None
    if epoch is not None :
        ckpt_prefix = os.path.join(ckpt_path, "Epoch-{}".format(epoch))
        initial_epoch = epoch

        for _file in ckpt_files :
            if _file.startswith(ckpt_prefix) :
                ckpt = os.path.splitext(_file)[0] # 확장자 부분 제외
                break

        if ckpt is None :
            raise FileNotFoundError("\n{} checkpoint를 찾을 수 없음\n".format(ckpt_prefix))

    else : # 가장 최근의 checkpoint를 불러옴
        ckpt = tf.train.latest_checkpoint(ckpt_path)
        initial_epoch = get_epoch(os.path.basename(ckpt)) # 파일 이름만 전달

    return ckpt, initial_epoch + 1

def load_checkpoint(**kwargs) :
    ckpt_model_path = kwargs["ckpt_path"]
    ckpt_epoch = kwargs["ckpt_epoch"]

    if os.path.isdir(ckpt_model_path) and not os.path.exists(ckpt_model_path) :
        raise FileNotFoundError("\n{} 경로를 찾을 수 없음\n".format(ckpt_model_path))

    if os.path.isdir(ckpt_model_path) : # Directory
        ckpt_model_path = os.path.join(ckpt_model_path, "ckpt")
        return find_checkpoint_file(ckpt_model_path, ckpt_epoch)
    else : # File
        return ckpt_model_path, get_epoch(ckpt_model_path) + 1
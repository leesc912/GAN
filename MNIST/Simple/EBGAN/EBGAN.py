from __future__ import print_function
from argparse import ArgumentParser, RawTextHelpFormatter
from Train import Trainer

parser = ArgumentParser(formatter_class = RawTextHelpFormatter)

parser.add_argument('-f', '--result_folder', type = str, default = None,
    help = "\n모델의 진행 사항을 저장할 폴더\n" + "default : 현재 위치에 Result folder 생성\n\n")

parser.add_argument('-P', '--ckpt_path', type = str, default = None,
    help = "\ncheckpoint path - default : None\n" + 
        "argument는 Train.py에서 folder 값 또는 checkpoint file name\n" +
        "ex1) -c ./foo/results/2019-04-18__004330\n" +
        "ex2) -c ./foo/results/2019-04-18__004330/ckpt.file\n\n")

parser.add_argument('-E', '--ckpt_epoch', type = int, default = None,
    help = "\ncheckpoint path가 folder일 경우 불러올 checkpoint의 epoch\n" +
        "만약 checkpoint의 path가 folder일 때, checkpoint_epoch를 설정하지 않으면\n" +
        "가장 최근의 checkpoint를 불러옴\n\n")

args                        = parser.parse_args()
kwargs                      = vars(args)

model = Trainer(**kwargs)
model.start()
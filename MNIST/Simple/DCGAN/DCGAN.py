from __future__ import print_function
from argparse import ArgumentParser, RawTextHelpFormatter
import os
from Model import DCGAN

parser = ArgumentParser(formatter_class = RawTextHelpFormatter)

parser.add_argument('-f', '--folder', type = str, default = os.path.join(os.getcwd(), "results"),
    help = "모델의 진행 사항을 저장할 폴더 - 기본값 : 현재 위치\n")

parser.add_argument('-p', '--ckpt_path', type = str, default = None,
    help = "checkpoint path - default : None\n" + 
        "argument는 Train.py에서 folder 값 또는 checkpoint file name\n" +
        "ex1) -c ./foo/results/2019-04-18__004330\n" +
        "ex2) -c ./foo/results/2019-04-18__004330/ckpt.file\n\n")

parser.add_argument('-E', '--ckpt_epoch', type = int, default = None,
    help = "checkpoint path가 folder일 경우 불러올 checkpoint의 epoch\n" +
        "만약 checkpoint의 path가 folder일 때, checkpoint_epoch를 설정하지 않으면\n" +
        "가장 최근의 checkpoint를 불러옴\n\n")

parser.add_argument('-i', '--interval', type = int, default = 1,
    help = "generator가 생성한 images를 저장하는 epoch 간격 - default : 1\n")

args                        = parser.parse_args()
kwargs                      = vars(args)

dcgan = DCGAN()
dcgan.train(**kwargs)
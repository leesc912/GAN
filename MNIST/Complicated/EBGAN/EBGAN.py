from __future__ import print_function
from type_check import bool_type_check
from argparse import ArgumentParser, RawTextHelpFormatter
from Train import Trainer

parser = ArgumentParser(formatter_class = RawTextHelpFormatter)

parser.add_argument('-t', '--test', type = bool_type_check, default = False,
    help = "\n테스트 모드. 테스트를 위한 checkpoint를 전달하지 않으면 error 발생\n" + "default : False\n\n")

parser.add_argument('-f', '--result_folder', type = str, default = None,
    help = "\n모델의 진행 사항을 저장할 폴더\n" + "default : 현재 위치에 Result folder 생성\n\n")

parser.add_argument('-e', '--epochs', type = int, default = 50000,
    help = "\ndefault : 50000\n\n")

parser.add_argument('-b', '--batch_size', type = int, default = 64,
    help = "\ndefault : 64\n\n")

parser.add_argument('-l', '--latent_size', type = int, default = 100,
    help = "\ndefault : 100\n\n")

parser.add_argument('-m', '--margin', type = int, default = 1,
    help = "\nmargin 값\n" + "default : 1\n\n")

parser.add_argument('-u', '--use_pt', type = bool_type_check, default = True,
    help = "\nGenerator에서 repelling regularizer사용\n" + "default : True\n\n")

parser.add_argument('-p', '--pt_weight', type = float, default = 0.1,
    help = "\nGenerator에서 구한 PT에 곱할 계수\n" + "default : 0.1\n\n")
    
parser.add_argument('-r', '--lr', type = float, default = 0.0002,
    help = "\nAdam optimizer의 learning rate\n" + "default : 0.0002\n\n")

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
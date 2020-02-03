from __future__ import print_function
from argparse_type_check import True_False_check
from argparse import ArgumentParser, RawTextHelpFormatter
import os
from Model import WGAN_GP

parser = ArgumentParser(formatter_class = RawTextHelpFormatter)

parser.add_argument('-t', '--test', type = True_False_check, default = False,
    help = "테스트 - 기본값 : False\n")

parser.add_argument('-f', '--folder', type = str, default = os.path.join(os.getcwd(), "results"),
    help = "모델의 진행 사항을 저장할 폴더 - 기본값 : 현재 위치\n")

parser.add_argument('-e', '--epochs', type = int, default = 50000,
    help = "에포크 수 - 기본값 : 50000\n")

parser.add_argument('-b', '--batch_size', type = int, default = 64,
    help = "batch size - default : 64\n")

parser.add_argument('-l', '--latent_size', type = int, default = 100,
    help = "latent size - default : 100\n")

parser.add_argument('-B', '--use_bias', type = True_False_check, default = False,
    help = "Generator에서 Conv 나 Dense layer에서 bias 사용 - default : False\n")

parser.add_argument('-U', '--use_layernorm', type = True_False_check, default = False,
    help = "Critic에서 LayerNormalization layer 적용 - default : False\n")

parser.add_argument('-d', '--dropout', type = float, default = 0,
    help = "0이면 Critic에 dropout layer를 적용하지 않음 - default : 0\n")

parser.add_argument('-s', '--slope', type = float, default = 0.2,
    help = "LeakyReLU slope - default : 0.2\n")

parser.add_argument('-n', '--n_critic', type = int, default = 5,
    help = "number of iterations of the critic per generator - default : 5\n")

parser.add_argument('-r', '--lr', nargs = '+', type = float, default = 0.0001,
    help = "learning rate of Adam optimizer\n" +
        "2개를 적는다면 각각 Generator와 Critic에 순서대로 할당됨\n"
        "Model을 불러온다면 사용하지 않음 - default : 0.0001\n")
    
parser.add_argument('--beta1', type = float, default = 0,
    help = "Adam optimizer의 Beta_1 값 - default : 0\n")

parser.add_argument('--beta2', type = float, default = 0.9,
    help = "Adam optimizer의 Beta_2 값 - default : 0.9\n")

parser.add_argument('-g', '--gp_coefficient', type = float, default = 10,
    help = "Gradient penalty coefficient - default : 10\n")

parser.add_argument('-p', '--ckpt_path', type = str, default = None,
    help = "Checkpoint path - default : None\n" + 
        "argument는 Train.py에서 folder 값 또는 checkpoint file name\n" +
        "ex1) -p ./foo/results/2019-04-18__004330\n" +
        "ex2) -p ./foo/results/2019-04-18__004330/ckpt.file\n\n")

parser.add_argument('-E', '--ckpt_epoch', type = int, default = None,
    help = "Checkpoint path가 folder일 경우 불러올 checkpoint의 epoch\n" +
        "만약 checkpoint의 path가 folder일 때, checkpoint_epoch를 설정하지 않으면\n" +
        "가장 최근의 checkpoint를 불러옴\n\n")

parser.add_argument('-i', '--interval', type = int, default = 1,
    help = "Generator가 생성한 image를 저장하는 epoch 간격 - default : 1\n")

args                        = parser.parse_args()
kwargs                      = vars(args)

wgan = WGAN_GP(**kwargs)
if kwargs["test"] :
    wgan.test(**kwargs)
else :
    wgan.train(**kwargs)
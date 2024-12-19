import argparse
import sys
import re

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def global_parser():

    parser = argparse.ArgumentParser(description='HighResMDE PyTorch implementation.', fromfile_prefix_chars='@')

    parser.add_argument('--file', type=str, help="Path to the file containing arguments")

    parser.add_argument('--pretrained_model', type=str, default=None)

    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--train_csv', type=str)
    parser.add_argument('--test_dir', type=str)
    parser.add_argument('--test_csv', type=str)

    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--model_size', type=str, default="large07")
    parser.add_argument('--encoder_grad',action=argparse.BooleanOptionalAction)
    parser.add_argument('--var_focus', type=float, default=0.85)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--lr_decay', type=float, default=0.95)
    parser.add_argument('--normal_blur', type=float, default=5.0)
    parser.add_argument('--total_epoch', type=int, default=50)
    parser.add_argument('--initial_epoch', type=int, default=5)
    parser.add_argument('--cutmix', action=argparse.BooleanOptionalAction)
    parser.add_argument('--cutflip', action=argparse.BooleanOptionalAction)
    parser.add_argument('--cut_prob', type=float, default=0.5)

    parser.add_argument('--loss_depth_weight', type=int, default=1)
    parser.add_argument('--loss_uncer_weight', type=int, default=1)
    parser.add_argument('--loss_normal_weight', type=int, default=5)
    parser.add_argument('--loss_dist_weight', type=float, default=0.25)
    parser.add_argument('--model_save_path', type=str, default="model.pth")

    parser.add_argument('--metric_cnt', type=int, default=9)
    parser.add_argument('--metric_save_path', type=str, default="metric.csv")

    def parse_args_from_file(file_path):
        # Read the arguments from the text file
        with open(file_path, 'r') as file:
            args = file.read().splitlines()  # Read each line as an argument
        return args

    args = parser.parse_args()

    if args.file:
        with open(args.file, 'r') as file:
            content = file.read()
            content = re.split(r'\s+', content.strip())  
            content = content + sys.argv[1:]
            args = parser.parse_args(content)
    
    print(args)

    return args

if __name__ == '__main__':
    args = global_parser()
    print(args)
    print(args.batch_size)
    print(args.train_csv)
    
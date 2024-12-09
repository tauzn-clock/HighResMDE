import argparse
import sys

def global_parser(current_file):

    parser = argparse.ArgumentParser(description='HighResMDE PyTorch implementation.')

    parser.add_argument('file', type=argparse.FileType('r'))

    parser.add_argument('--pretrained_model', type=str, default=None)

    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--train_csv', type=str)
    parser.add_argument('--test_dir', type=str)
    parser.add_argument('--test_csv', type=str)

    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--model_size', type=str, default="large07")
    parser.add_argument('--swinv2_specific_path', type=str, default=None)
    parser.add_argument('--var_focus', type=float, default=0.85)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--lr_decay', type=float, default=0.95)

    parser.add_argument('--loss_depth_weight', type=int, default=1)
    parser.add_argument('--loss_uncer_weight', type=int, default=1)
    parser.add_argument('--loss_normal_weight', type=int, default=5)
    parser.add_argument('--loss_dist_weight', type=float, default=0.25)
    parser.add_argument('--model_save_path', type=str, default="model.pth")

    parser.add_argument('--metric_cnt', type=int, default=9)
    parser.add_argument('--metric_save_path', type=str, default="metric.csv")

    #args = parser.parse_args([f'@{current_file}'])

    args = parser.parse_args([current_file])

    return args

if __name__ == '__main__':
    args = global_parser(sys.argv[1])
    print(args)
    print(args.batch_size)
    print(args.train_csv)
    
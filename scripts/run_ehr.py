import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from exp.exp_ehr import Exp_EHR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='sampledata', help='dataset root folder')
    parser.add_argument('--ehr_task', type=str, choices=['ihm', 'pheno'], required=True)
    parser.add_argument('--llm_model_path', type=str, default='JackFram/llama-160m')
    parser.add_argument('--huggingface_token', type=str, default=None)
    parser.add_argument('--max_seq_len', type=int, default=256,
                        help='max sequence length for tokenizer')
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_ff', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--patch_len', type=int, default=16,
                        help='patch length for PatchTST')
    parser.add_argument('--stride', type=int, default=8,
                        help='patch stride for PatchTST')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--train_epochs', type=int, default=1)
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--use_multi_gpu', action='store_true')
    parser.add_argument('--devices', type=str, default='0')
    args = parser.parse_args()

    if args.use_gpu and args.use_multi_gpu:
        args.device_ids = [int(x) for x in args.devices.split(',')]
        args.gpu = args.device_ids[0]
    else:
        args.gpu = 0
    exp = Exp_EHR(args)
    setting = f"ehr_{args.ehr_task}"
    exp.train(setting)
    exp.test(setting)


if __name__ == '__main__':
    main()

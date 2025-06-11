import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from exp.exp_ehr import Exp_EHR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/home/ubuntu/hcy50662/output_mimic3', help='dataset root folder')
    parser.add_argument('--ehr_task', type=str, default='pheno', choices=['ihm', 'pheno'])
    parser.add_argument('--llm_model_path', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--huggingface_token', type=str, default="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")
    parser.add_argument('--max_seq_len', type=int, default=5000,
                        help='max sequence length for tokenizer')
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_ff', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--patch_len', type=int, default=4,
                        help='patch length for PatchTST')
    parser.add_argument('--stride', type=int, default=2,
                        help='patch stride for PatchTST')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--train_epochs', type=int, default=5000)
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--use_multi_gpu', action='store_true')
    parser.add_argument('--devices', type=str, default='0')
    
    # wandb相关参数
    parser.add_argument('--use_wandb', action='store_true', help='是否使用wandb记录实验')
    parser.add_argument('--wandb_project', type=str, default='MM-TSFlib-EHR', help='wandb项目名称')
    parser.add_argument('--exp_name', type=str, default=None, help='实验名称，用于wandb标识')
    
    args = parser.parse_args()

    if args.use_gpu and args.use_multi_gpu:
        args.device_ids = [int(x) for x in args.devices.split(',')]
        args.gpu = args.device_ids[0]
    else:
        args.gpu = 0
        
    # 如果没有指定实验名称，自动生成一个
    if args.exp_name is None:
        args.exp_name = f"{args.ehr_task}_d{args.d_model}_h{args.n_heads}_l{args.e_layers}_lr{args.learning_rate}"
    
    exp = Exp_EHR(args)
    setting = f"ehr_{args.ehr_task}"
    
    print(f"开始训练 - 任务: {args.ehr_task}")
    if args.use_wandb:
        print(f"使用wandb记录实验 - 项目: {args.wandb_project}, 实验名: {args.exp_name}")
    
    exp.train(setting)
    exp.test(setting)


if __name__ == '__main__':
    main()

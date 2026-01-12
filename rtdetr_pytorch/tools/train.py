"""by lyuwenyu
"""

import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse
import torch
import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS


def main(args, ) -> None:
    '''main
    '''
    dist.init_distributed()
    if args.seed is not None:
        dist.set_seed(args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    # # 1. 先创建一个字典，放那些肯定要传的参数
    # cfg_kwargs = {
    #     'resume': args.resume,
    #     'use_amp': args.amp,
    #     'tuning': args.tuning
    # }
    # # 2. 【关键一步】只有当命令行真的指定了 output-dir 时，才把它加进去
    # # 如果 args.output_dir 是 None，这里就不传，YAMLConfig 就会自动读取 .yml 里的配置
    # if args.output_dir is not None:
    #     cfg_kwargs['output_dir'] = args.output_dir

    cfg = YAMLConfig(
        args.config,
        resume=args.resume,
        use_amp=args.amp,
        tuning=args.tuning,
    )
    # # 3. 使用 **cfg_kwargs 把参数解包传进去
    # cfg = YAMLConfig(
    #     args.config,
    #     **cfg_kwargs
    # )

    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    if args.test_only:
        solver.val()
    else:
        solver.fit()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default="../configs/rtdetr/rtdetr_r50vd_6x_coco.yml")
    # parser.add_argument('--resume', '-r', type=str, default="../tools/output/rtdetr_r50vd_6x_coco/base/checkpoint0048.pth")
    # parser.add_argument('--resume', '-r', type=str, default="../tools/output/rtdetr_r50vd_6x_coco/small_obj/checkpoint0000.pth")
    # parser.add_argument('--resume', '-r', type=str,
    #                     default="../tools/output/rtdetr_r50vd_6x_coco/query_select/checkpoint0071.pth")
    parser.add_argument('--resume', '-r', type=str,)
    parser.add_argument('--tuning', '-t', type=str, )
    parser.add_argument('--test-only', action='store_true', default=False)
    parser.add_argument('--amp', action='store_true', default=False,)
    parser.add_argument('--seed', type=int, help='seed',)
    parser.add_argument('--epoches', type=int, default=10)
    # parser.add_argument('--output-dir', type=str, default=None, help='output directory')
    # parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    main(args)

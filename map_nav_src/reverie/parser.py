import argparse
import os


def parse_args():

    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--root_dir', type=str, default='/root/mount/VLN-DUET/datasets')#数据集路径
    parser.add_argument('--dataset', type=str, default='reverie', choices=['reverie'])
    parser.add_argument('--output_dir', type=str, default='/root/mount/VLN-DUET/datasets/REVERIE/run_test', help='experiment id')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tokenizer', choices=['bert', 'xlm'], default='bert')

    parser.add_argument('--fusion', choices=['global', 'local', 'avg', 'dynamic'], default='dynamic')
    parser.add_argument('--dagger_sample', choices=['sample', 'expl_sample', 'argmax'], default='sample')
    parser.add_argument('--expl_max_ratio', type=float, default=0.6)
    parser.add_argument('--loss_nav_3', action='store_true', default=False)

    # distributional training (single-node, multiple-gpus)
    parser.add_argument('--world_size', type=int, default=1, help='number of gpus')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument("--node_rank", type=int, default=0, help="Id of the node")
    
    # General
    parser.add_argument('--iters', type=int, default=50000, help='training iterations')
    parser.add_argument('--log_every', type=int, default=1000)
    parser.add_argument('--eval_first', action='store_true', default=False)

    # Data preparation
    parser.add_argument('--max_instr_len', type=int, default=200)
    parser.add_argument('--max_action_len', type=int, default=15)
    parser.add_argument('--max_objects', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--ignoreid', type=int, default=-100, help='ignoreid for action')
    
    # Load the model from
    parser.add_argument("--resume_file", default='/root/mount/VLN-DUET/datasets/best_val_unseen', help='path of the trained model')
    parser.add_argument("--resume_optimizer", action="store_true", default=False)

    # Augmented Paths from
    parser.add_argument("--multi_endpoints", default=True, action="store_true")
    parser.add_argument("--multi_startpoints", default=False, action="store_true")
    parser.add_argument("--aug_only", default=False, action="store_true")
    parser.add_argument("--aug", default=None)
    parser.add_argument('--bert_ckpt_file', default='/root/mount/VLN-DUET/datasets/REVERIE/model_step_84000.pt', help='init vlnbert')

    # Listener Model Config
    parser.add_argument("--ml_weight", type=float, default=0.20)
    parser.add_argument('--entropy_loss_weight', type=float, default=0.01)

    parser.add_argument("--features", type=str, default='vitbase')
    parser.add_argument('--obj_features', type=str, default='vitbase')

    parser.add_argument('--fix_lang_embedding', action='store_true', default=True)
    parser.add_argument('--fix_pano_embedding', action='store_true', default=False)
    parser.add_argument('--fix_local_branch', action='store_true', default=True)
    parser.add_argument('--fix_global_branch', action='store_true', default=True)
    parser.add_argument('--fix_glocal_fuse', action='store_true', default=True)
    parser.add_argument('--og', action='store_true', default=True)

    parser.add_argument('--num_l_layers', type=int, default=9)
    parser.add_argument('--num_pano_layers', type=int, default=2)
    parser.add_argument('--num_x_layers', type=int, default=4)

    parser.add_argument('--enc_full_graph', default=True, action='store_true')
    parser.add_argument('--graph_sprels', action='store_true', default=True)

    parser.add_argument('--record_rt', action='store_true', default=False)

    # Dropout Param
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--feat_dropout', type=float, default=0.4)

    # Submision configuration
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument("--submit", action='store_true', default=False)
    parser.add_argument('--no_backtrack', action='store_true', default=False)
    parser.add_argument('--detailed_output', action='store_true', default=False)

    # Training Configurations
    parser.add_argument(
        '--optim', type=str, default='adamW',
        choices=['rms', 'adam', 'adamW', 'sgd']
    )    # rms, adam
    parser.add_argument('--lr', type=float, default=1e-5, help="the learning rate")
    parser.add_argument('--decay', dest='weight_decay', type=float, default=0.)
    parser.add_argument(
        '--feedback', type=str, default='sample',
        help='How to choose next position, one of ``teacher``, ``sample`` and ``argmax``'
    )
    parser.add_argument('--epsilon', type=float, default=0.1, help='')
    parser.add_argument('--progress', action='store_true', default=False)
    parser.add_argument('--cand', action='store_true', default=False)
    parser.add_argument('--use_history', action='store_true', default=True)
    parser.add_argument('--condition', action='store_true', default=True)
    parser.add_argument('--full_history', action='store_true', default=True)
    parser.add_argument('--hypamas_a', type=float, default=0.5)
    # parser.add_argument('--method', type=str, default='method1')
    # parser.add_argument('--warmup_epochs', type=int, default=10)
    # parser.add_argument('--epochs', type=int, default=20)
    # parser.add_argument('--lr_scheduler', default='warmup', type=str)
    # parser.add_argument('--init_lr', type=float, default=0.01)
    # parser.add_argument('--optim_select', action='store_true', default=False)


    # Model hyper params:
    parser.add_argument("--angle_feat_size", type=int, default=4)
    parser.add_argument('--image_feat_size', type=int, default=768)
    parser.add_argument('--obj_feat_size', type=int, default=768)
    parser.add_argument('--views', type=int, default=36)

    # # A2C
    parser.add_argument("--gamma", default=0., type=float, help='reward discount factor')
    parser.add_argument(
        "--normalize", dest="normalize_loss", default="total", 
        type=str, help='batch or total'
    )
    parser.add_argument('--train_alg', 
        choices=['imitation', 'dagger'], 
        default='dagger'
    )

    # PGN
    parser.add_argument('--pgn_length', default=42, type=int)
    parser.add_argument('--prompt_mode', default='pgn', type=str)
    parser.add_argument('--pgn_mixture_size', default=256, type=int)
    parser.add_argument('--pgn_act_fn', default='sigmoid', type=str)

    # PGN Model
    parser.add_argument('--pgn_model_type', default='resnet10', type=str)
    parser.add_argument('--pgn_proj_type', default='linear', type=str)
    parser.add_argument('--pgn_resolution', default=224, type=int)
    parser.add_argument('--nr_groups', default=4, type=int)
    parser.add_argument('--blocks_per_group', default=1, type=int)
    parser.add_argument('--initial_channels', default=16, type=int)
    parser.add_argument('--init_max_pool', action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument('--disable_pgn', action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument('--vision_model_type', default='vit',
                        type=str)
    parser.add_argument('--pretrained_pgn', action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument('--size', default=36)

    args, _ = parser.parse_known_args()

    args = postprocess_args(args)

    return args


def postprocess_args(args):
    ROOTDIR = args.root_dir

    # Setup input paths
    ft_file_map = {
        'vitbase': 'pth_vit_base_patch16_224_imagenet.hdf5',
    }
    args.img_ft_file = os.path.join(ROOTDIR, 'R2R', 'features', ft_file_map[args.features])

    obj_ft_file_map = {
        'vitbase': 'obj.avg.top3.min80_vit_base_patch16_224_imagenet.hdf5',
    }
    args.obj_ft_file = os.path.join(ROOTDIR, 'REVERIE', 'features', obj_ft_file_map[args.obj_features])
    
    args.connectivity_dir = os.path.join(ROOTDIR, 'R2R', 'connectivity')
    args.scan_data_dir = os.path.join(ROOTDIR, 'Matterport3D', 'v1_unzip_scans')

    args.anno_dir = os.path.join(ROOTDIR, 'REVERIE', 'annotations')
    args.anno_pretrain_dir = os.path.join(ROOTDIR, 'REVERIE', 'annotations', 'pretrain')

    # Build paths
    args.ckpt_dir = os.path.join(args.output_dir, 'ckpts')
    args.log_dir = os.path.join(args.output_dir, 'logs')
    args.pred_dir = os.path.join(args.output_dir, 'preds')

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.pred_dir, exist_ok=True)

    return args


import torch


def get_tokenizer(args):
    from transformers import AutoTokenizer
    if args.tokenizer == 'xlm':
        cfg_name = 'xlm-roberta-base'
    else:
        cfg_name = 'bert-base-uncased'
        path = '/root/mount/VLN-DUET/bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(path)
    return tokenizer

def get_vlnbert_models(args, config=None):
    
    from transformers import PretrainedConfig
    from models.vilmodel import GlocalTextPathNavCMT
    
    visual_embedding_dims = {
    'vit': 768,
    'dino': 384
    }
    model_name_or_path = args.bert_ckpt_file
    new_ckpt_weights = {}
    if model_name_or_path is not None:
        ckpt_weights = torch.load(model_name_or_path)
        for k, v in ckpt_weights.items():
            if k.startswith('module'):
                k = k[7:]    
            if '_head' in k or 'sap_fuse' in k:
                new_ckpt_weights['bert.' + k] = v
            else:
                new_ckpt_weights[k] = v
            
    if args.tokenizer == 'xlm':
        cfg_name = 'xlm-roberta-base'
    else:
        cfg_name = 'bert-base-uncased'
    vis_config = PretrainedConfig.from_pretrained(cfg_name)

    if args.tokenizer == 'xlm':
        vis_config.type_vocab_size = 2
    
    vis_config.max_action_steps = 100
    vis_config.image_feat_size = args.image_feat_size
    vis_config.angle_feat_size = args.angle_feat_size
    vis_config.obj_feat_size = args.obj_feat_size
    vis_config.obj_loc_size = 3
    vis_config.num_l_layers = args.num_l_layers
    vis_config.num_pano_layers = args.num_pano_layers
    vis_config.num_x_layers = args.num_x_layers
    vis_config.graph_sprels = args.graph_sprels
    vis_config.glocal_fuse = args.fusion == 'dynamic'
    vis_config.progress = args.progress

    vis_config.fix_lang_embedding = args.fix_lang_embedding
    vis_config.fix_pano_embedding = args.fix_pano_embedding
    vis_config.fix_local_branch = args.fix_local_branch
    vis_config.fix_global_branch = args.fix_global_branch
    vis_config.fix_glocal_fuse = args.fix_glocal_fuse
    vis_config.og = args.og

    vis_config.update_lang_bert = not args.fix_lang_embedding
    vis_config.output_attentions = True
    vis_config.pred_head_dropout_prob = 0.1
    vis_config.use_lang2visn_attn = args.progress
    vis_config.pgn_settings = {}
    if not args.disable_pgn:
        pgn_settings = {
            'prompt_mode': args.prompt_mode,
            'nr_output_vectors': args.pgn_length,
            'vector_dim': visual_embedding_dims[args.vision_model_type],
            'mixture_size': args.pgn_mixture_size,
            'pretrained_pgn': args.pretrained_pgn,
            'model_type': args.pgn_model_type,
            'proj_type': args.pgn_proj_type,
            'pgn_act_fn': args.pgn_act_fn,
            'nr_groups': args.nr_groups,
            'blocks_per_group': args.blocks_per_group,
            'initial_channels': args.initial_channels,
            'init_max_pool': args.init_max_pool
        }
    else:
        pgn_settings = None
    vis_config.pgn_settings = pgn_settings

    visual_model = GlocalTextPathNavCMT.from_pretrained(
        pretrained_model_name_or_path=None, 
        config=vis_config, 
        state_dict=new_ckpt_weights)
        
    return visual_model

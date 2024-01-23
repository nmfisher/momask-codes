from typing import Any

import argparse
import os
import torch
from cog import BasePredictor, Input, Path, File
from os.path import join as pjoin

import torch.nn.functional as F

from models.mask_transformer.transformer import MaskTransformer, ResidualTransformer
from models.vq.model import RVQVAE, LengthEstimator

from options.eval_option import EvalT2MOptions
from utils.get_opt import get_opt

from utils.fixseed import fixseed
from visualization.joints2bvh import Joint2BVHConvertor
from torch.distributions.categorical import Categorical

from utils.motion_process import recover_from_ric
from utils.plot_script import plot_3d_motion

from utils.paramUtil import t2m_kinematic_chain
import uuid
import numpy as np
import tempfile

clip_version = 'ViT-B/32'

os.environ['PYOPENGL_PLATFORM'] = 'egl'
def load_vq_model(vq_opt):
    # opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')
    vq_model = RVQVAE(vq_opt,
                vq_opt.dim_pose,
                vq_opt.nb_code,
                vq_opt.code_dim,
                vq_opt.output_emb_width,
                vq_opt.down_t,
                vq_opt.stride_t,
                vq_opt.width,
                vq_opt.depth,
                vq_opt.dilation_growth_rate,
                vq_opt.vq_act,
                vq_opt.vq_norm)
    ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', 'net_best_fid.tar'),
                            map_location='cpu')
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {vq_opt.name} Completed!')
    return vq_model, vq_opt

def load_trans_model(model_opt, opt, which_model):
    t2m_transformer = MaskTransformer(code_dim=model_opt.code_dim,
                                      cond_mode='text',
                                      latent_dim=model_opt.latent_dim,
                                      ff_size=model_opt.ff_size,
                                      num_layers=model_opt.n_layers,
                                      num_heads=model_opt.n_heads,
                                      dropout=model_opt.dropout,
                                      clip_dim=512,
                                      cond_drop_prob=model_opt.cond_drop_prob,
                                      clip_version=clip_version,
                                      opt=model_opt)
    ckpt = torch.load(pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, model_opt.name, 'model', which_model),
                      map_location='cpu')
    model_key = 't2m_transformer' if 't2m_transformer' in ckpt else 'trans'
    # print(ckpt.keys())
    missing_keys, unexpected_keys = t2m_transformer.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Transformer {opt.name} from epoch {ckpt["ep"]}!')
    return t2m_transformer

def load_res_model(res_opt, vq_opt, opt):
    res_opt.num_quantizers = vq_opt.num_quantizers
    res_opt.num_tokens = vq_opt.nb_code
    res_transformer = ResidualTransformer(code_dim=vq_opt.code_dim,
                                            cond_mode='text',
                                            latent_dim=res_opt.latent_dim,
                                            ff_size=res_opt.ff_size,
                                            num_layers=res_opt.n_layers,
                                            num_heads=res_opt.n_heads,
                                            dropout=res_opt.dropout,
                                            clip_dim=512,
                                            shared_codebook=vq_opt.shared_codebook,
                                            cond_drop_prob=res_opt.cond_drop_prob,
                                            # codebook=vq_model.quantizer.codebooks[0] if opt.fix_token_emb else None,
                                            share_weight=res_opt.share_weight,
                                            clip_version=clip_version,
                                            opt=res_opt)

    ckpt = torch.load(pjoin(res_opt.checkpoints_dir, res_opt.dataset_name, res_opt.name, 'model', 'net_best_fid.tar'),
                      map_location=opt.device)
    missing_keys, unexpected_keys = res_transformer.load_state_dict(ckpt['res_transformer'], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Residual Transformer {res_opt.name} from epoch {ckpt["ep"]}!')
    return res_transformer

def load_len_estimator(opt):
    model = LengthEstimator(512, 50)
    ckpt = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, 'length_estimator', 'model', 'finest.tar'),
                      map_location=opt.device)
    model.load_state_dict(ckpt['estimator'])
    print(f'Loading Length Estimator from epoch {ckpt["epoch"]}!')
    return model

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        parser = EvalT2MOptions()
        opt = parser.parse()
        fixseed(opt.seed)

        opt.device = "cuda:0" #torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
        
        print(f"Creating Predictor with device {opt.device}")
        print(opt)
        torch.autograd.set_detect_anomaly(True)

        dim_pose = 251 if opt.dataset_name == 'kit' else 263

        # out_dir = pjoin(opt.check)
        root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
        model_dir = pjoin(root_dir, 'model')
        result_dir = pjoin('./generation', opt.ext)
        joints_dir = pjoin(result_dir, 'joints')
        animation_dir = pjoin(result_dir, 'animations')
        os.makedirs(joints_dir, exist_ok=True)
        os.makedirs(animation_dir,exist_ok=True)

        model_opt_path = pjoin(root_dir, 'opt.txt')
        model_opt = get_opt(model_opt_path, device=opt.device)


        #######################
        ######Loading RVQ######
        #######################
        vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'opt.txt')
        vq_opt = get_opt(vq_opt_path, device=opt.device)
        vq_opt.dim_pose = dim_pose
        vq_model, vq_opt = load_vq_model(vq_opt)
        self.vq_model = vq_model
        self.vq_opt = vq_opt

        model_opt.num_tokens = vq_opt.nb_code
        model_opt.num_quantizers = vq_opt.num_quantizers
        model_opt.code_dim = vq_opt.code_dim

        #################################
        ######Loading R-Transformer######
        #################################
        res_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.res_name, 'opt.txt')
        res_opt = get_opt(res_opt_path, device=opt.device)
        res_model = load_res_model(res_opt, vq_opt, opt)
        self.res_model = res_model

        assert res_opt.vq_name == model_opt.vq_name

        #################################
        ######Loading M-Transformer######
        #################################
        t2m_transformer = load_trans_model(model_opt, opt, 'latest.tar')
        self.t2m_transformer = t2m_transformer

        ##################################
        #####Loading Length Predictor#####
        ##################################
        length_estimator = load_len_estimator(model_opt)
        self.length_estimator = length_estimator

        t2m_transformer.eval()
        vq_model.eval()
        res_model.eval()
        length_estimator.eval()

        res_model.to(opt.device)
        t2m_transformer.to(opt.device)
        vq_model.to(opt.device)
        length_estimator.to(opt.device)

        ##### ---- Dataloader ---- #####
        opt.nb_joints = 21 if opt.dataset_name == 'kit' else 22

        self.mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'mean.npy'))
        self.std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'std.npy'))
        self.opt = opt

    def inv_transform(self, data):
        return data * self.std + self.mean

    # Define the arguments and types the model takes as input
    def predict(self,
                prompt: str = Input(description="Prompt used to generate animation"),
    ) -> str:
        prompt_list = [prompt]
        length_list = []
        est_length = True

        if est_length:
            text_embedding = self.t2m_transformer.encode_text(prompt_list)
            pred_dis = self.length_estimator(text_embedding)
            probs = F.softmax(pred_dis, dim=-1)  # (b, ntoken)
            token_lens = Categorical(probs).sample()  # (b, seqlen)
            # lengths = torch.multinomial()
        else:
            token_lens = torch.LongTensor(length_list) // 4
            token_lens = token_lens.to(self.opt.device).long()
        print(f"token_lens {token_lens.size()}")
        m_length = token_lens * 4
        captions = prompt_list

        sample = 0
        kinematic_chain = t2m_kinematic_chain
        converter = Joint2BVHConvertor()

        for r in range(self.opt.repeat_times):
            with torch.no_grad():
                mids = self.t2m_transformer.generate(captions, token_lens,
                                                timesteps=self.opt.time_steps,
                                                cond_scale=self.opt.cond_scale,
                                                temperature=self.opt.temperature,
                                                topk_filter_thres=self.opt.topkr,
                                                gsample=self.opt.gumbel_sample)
                mids = self.res_model.generate(mids, captions, token_lens, temperature=1, cond_scale=5)
                pred_motions = self.vq_model.forward_decoder(mids)
                pred_motions = pred_motions.detach().cpu().numpy()
                data = self.inv_transform(pred_motions)
            for k, (caption, joint_data)  in enumerate(zip(captions, data)):
                joint_data = joint_data[:m_length[k]]
                joint = recover_from_ric(torch.from_numpy(joint_data).float(), 22).numpy()
                
                with tempfile.NamedTemporaryFile() as outfile:
                    _, joint = converter.convert(joint, filename=outfile.name, iterations=100, foot_ik=False)
                    with open(outfile.name, "r") as infile:
                        return infile.read()

#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat_baselines.rl.ppo.ppo import PPONonOracle
import json
import os
import time
import pathlib
import sys
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

from einops import rearrange
from matplotlib import pyplot as plt
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch_scatter
import tqdm
from torch.optim.lr_scheduler import LambdaLR

from habitat import Config
from habitat.core.logging import logger
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.common.base_trainer import BaseRLTrainerNonOracle, BaseRLTrainerOracle
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.rollout_storage import RolloutStorageOracle, RolloutStorageNonOracle
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.common.utils import (
    batch_obs,
    generate_video,
    linear_decay,
)
from habitat_baselines.rl.ppo import PPONonOracle, PPOOracle, BaselinePolicyNonOracle, BaselinePolicyOracle, ProposedPolicyOracle
from utils.log_manager import LogManager
from utils.log_writer import LogWriter
from habitat.utils.visualizations import fog_of_war, maps


def to_grid(coordinate_min, coordinate_max, global_map_size, position):
    grid_size = (coordinate_max - coordinate_min) / global_map_size
    grid_x = ((coordinate_max - position[0]) / grid_size).round()
    grid_y = ((position[1] - coordinate_min) / grid_size).round()
    return int(grid_x), int(grid_y)


def draw_projection(image, depth, s, global_map_size, coordinate_min, coordinate_max):
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)
    depth = torch.tensor(depth).permute(2, 0, 1).unsqueeze(0)
    spatial_locs, valid_inputs = _compute_spatial_locs(depth, s, global_map_size, coordinate_min, coordinate_max)
    x_gp1 = _project_to_ground_plane(image, spatial_locs, valid_inputs, s)
    
    return x_gp1


def _project_to_ground_plane(img_feats, spatial_locs, valid_inputs, s):
    outh, outw = (s, s)
    bs, f, HbyK, WbyK = img_feats.shape
    device = img_feats.device
    eps=-1e16
    K = 1

    # Sub-sample spatial_locs, valid_inputs according to img_feats resolution.
    idxes_ss = ((torch.arange(0, HbyK, 1)*K).long().to(device), \
                (torch.arange(0, WbyK, 1)*K).long().to(device))

    spatial_locs_ss = spatial_locs[:, :, idxes_ss[0][:, None], idxes_ss[1]] # (bs, 2, HbyK, WbyK)
    valid_inputs_ss = valid_inputs[:, :, idxes_ss[0][:, None], idxes_ss[1]] # (bs, 1, HbyK, WbyK)
    valid_inputs_ss = valid_inputs_ss.squeeze(1) # (bs, HbyK, WbyK)
    invalid_inputs_ss = ~valid_inputs_ss

    # Filter out invalid spatial locations
    invalid_spatial_locs = (spatial_locs_ss[:, 1] >= outh) | (spatial_locs_ss[:, 1] < 0 ) | \
                        (spatial_locs_ss[:, 0] >= outw) | (spatial_locs_ss[:, 0] < 0 ) # (bs, H, W)

    invalid_writes = invalid_spatial_locs | invalid_inputs_ss

    # Set the idxes for all invalid locations to (0, 0)
    spatial_locs_ss[:, 0][invalid_writes] = 0
    spatial_locs_ss[:, 1][invalid_writes] = 0

    # Weird hack to account for max-pooling negative feature values
    invalid_writes_f = rearrange(invalid_writes, 'b h w -> b () h w').float()
    img_feats_masked = img_feats * (1 - invalid_writes_f) + eps * invalid_writes_f
    img_feats_masked = rearrange(img_feats_masked, 'b e h w -> b e (h w)')

    # Linearize ground-plane indices (linear idx = y * W + x)
    linear_locs_ss = spatial_locs_ss[:, 1] * outw + spatial_locs_ss[:, 0] # (bs, H, W)
    linear_locs_ss = rearrange(linear_locs_ss, 'b h w -> b () (h w)')
    linear_locs_ss = linear_locs_ss.expand(-1, f, -1) # .contiguous()

    proj_feats, _ = torch_scatter.scatter_max(
                        img_feats_masked,
                        linear_locs_ss,
                        dim=2,
                        dim_size=outh*outw,
                    )
    proj_feats = rearrange(proj_feats, 'b e (h w) -> b e h w', h=outh)

    # Replace invalid features with zeros
    eps_mask = (proj_feats == eps).float()
    proj_feats = proj_feats * (1 - eps_mask) + eps_mask * (proj_feats - eps)

    return proj_feats


def _compute_spatial_locs(depth_inputs, s, global_map_size, coordinate_min, coordinate_max):
    bs, _, imh, imw = depth_inputs.shape
    local_scale = float(coordinate_max - coordinate_min)/float(global_map_size)
    cx, cy = 256./2., 256./2.
    fx = fy =  (256. / 2.) / np.tan(np.deg2rad(79. / 2.))

    #2D image coordinates
    x    = rearrange(torch.arange(0, imw), 'w -> () () () w')
    y    = rearrange(torch.arange(imh, 0, step=-1), 'h -> () () h ()')
    xx   = (x - cx) / fx
    yy   = (y - cy) / fy

    # 3D real-world coordinates (in meters)
    Z            = depth_inputs
    X            = xx * Z
    Y            = yy * Z
    # valid_inputs = (depth_inputs != 0) & ((Y < 1) & (Y > -1))
    valid_inputs = (depth_inputs != 0) & ((Y > -0.5) & (Y < 1))

    # 2D ground projection coordinates (in meters)
    # Note: map_scale - dimension of each grid in meters
    # - depth/scale + (s-1)/2 since image convention is image y downward
    # and agent is facing upwards.
    x_gp            = ( (X / local_scale) + (s-1)/2).round().long() # (bs, 1, imh, imw)
    y_gp            = (-(Z / local_scale) + (s-1)/2).round().long() # (bs, 1, imh, imw)

    return torch.cat([x_gp, y_gp], dim=1), valid_inputs


def rotate_tensor(x_gp, heading):
    sin_t = torch.sin(heading.squeeze(1))
    cos_t = torch.cos(heading.squeeze(1))
    A = torch.zeros(x_gp.size(0), 2, 3)
    A[:, 0, 0] = cos_t
    A[:, 0, 1] = sin_t
    A[:, 1, 0] = -sin_t
    A[:, 1, 1] = cos_t

    grid = F.affine_grid(A, x_gp.size())
    rotated_x_gp = F.grid_sample(x_gp, grid)
    return rotated_x_gp


@baseline_registry.register_trainer(name="oracle2")
class PPOTrainerO2(BaseRLTrainerOracle):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["Nav-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None
        if config is not None:
            logger.info(f"config: {config}")
            
        logger.info("THIS IS PPO TRAINER 2!!!")

        self._static_encoder = False
        self._encoder = None
        
        self._num_picture = config.TASK_CONFIG.TASK.PICTURE.NUM_PICTURE
        #撮った写真のRGB画像を保存
        self._taken_picture = []
        #撮った写真のciと位置情報、向きを保存
        self._taken_picture_list = []
        
        # with open('mapDictFull.pickle', 'rb') as handle:
            # self.mapCache = pickle.load(handle)
            
    def _do_take_picture_object(self, top_down_map, fog_of_war_map):
        # maps.MAP_TARGET_POINT_INDICATOR(6)が写真の中にあればtrueを返す
        for i in range(len(top_down_map)):
            for j in range(len(top_down_map[0])):
                if top_down_map[i][j] == maps.MAP_TARGET_POINT_INDICATOR:
                    if fog_of_war_map[i][j] == 1:
                        return True
                        
        return False

    def _setup_actor_critic_agent(self, ppo_cfg: Config) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        self.actor_critic = ProposedPolicyOracle(
            agent_type = self.config.TRAINER_NAME,
            observation_space=self.envs.observation_spaces[0],
            action_space=self.envs.action_spaces[0],
            hidden_size=ppo_cfg.hidden_size,
            device=self.device,
            previous_action_embedding_size=self.config.RL.PREVIOUS_ACTION_EMBEDDING_SIZE,
            use_previous_action=self.config.RL.PREVIOUS_ACTION
        )
        self.actor_critic.to(self.device)

        self.agent = PPOOracle(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            use_normalized_advantage=ppo_cfg.use_normalized_advantage,
        )

    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "state_dict": self.agent.state_dict(),
            "config": self.config,
        }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision", "traj_metrics"}

    @classmethod
    def _extract_scalars_from_info(
        cls, info: Dict[str, Any]
    ) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if k in cls.METRICS_BLACKLIST:
                continue

            if k == "ci":
                result[k] = float(v[0])
                
            elif isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(
                            v
                        ).items()
                        if (k + "." + subk) not in cls.METRICS_BLACKLIST
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)

        return result

    @classmethod
    def _extract_scalars_from_infos(
        cls, infos: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:

        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in cls._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)

        return results
        

    def _collect_rollout_step(
        self, rollouts, current_episode_reward, current_episode_exp_area, running_episode_stats
    ):
        pth_time = 0.0
        env_time = 0.0

        t_sample_action = time.time()
        # sample actions
        with torch.no_grad():
            step_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }

            (
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states,
            ) = self.actor_critic.act(
                step_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
            )

        pth_time += time.time() - t_sample_action

        t_step_env = time.time()

        outputs = self.envs.step([a[0].item() for a in actions])
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

        env_time += time.time() - t_step_env

        t_update_stats = time.time()
        batch = batch_obs(observations, device=self.device)
        
        reward = []
        ci = []
        exp_area = [] # 探索済みのエリア()
        exp_area_pre = []
        matrics = []
        fog_of_war_map = []
        top_down_map = [] 
        n_envs = self.envs.num_envs
        for i in range(n_envs):
            reward.append(rewards[i][0][0])
            ci.append(rewards[i][0][1])
            exp_area.append(rewards[i][0][2])
            exp_area_pre.append(rewards[i][0][3])
            matrics.append(rewards[i][1])
            fog_of_war_map.append(infos[i]["picture_range_map"]["fog_of_war_mask"])
            top_down_map.append(infos[i]["picture_range_map"]["map"])
        
            
        for n in range(len(observations)):
            #TAKE_PICTUREが呼び出されたかを検証
            if ci[n] != -sys.float_info.max:
                # 今回撮ったpicture(p_n)が保存してあるpicture(p_k)とかぶっているkを保存
                cover_list = [] 
                picture_range_map = self._create_picture_range_map(top_down_map[n], fog_of_war_map[n])
                
                is_ok = self._do_take_picture_object(top_down_map[n], fog_of_war_map[n])
                if is_ok == True:
                    ci[n] += 100
                
                self._take_picture += 1
                # p_kのそれぞれのpicture_range_mapのリスト
                pre_fog_of_war_map = [sublist[2] for sublist in self._taken_picture_list[n]]
                    
                # それぞれと閾値より被っているか計算
                idx = -1
                min_ci = ci[n]
                for k in range(len(pre_fog_of_war_map)):
                    # 閾値よりも被っていたらcover_listにkを追加
                    if self._check_percentage_of_fog(picture_range_map, pre_fog_of_war_map[k]) == True:
                        cover_list.append(k)
                            
                    #ciの最小値の写真を探索(１つも被っていない時用)
                    if min_ci < self._taken_picture_list[n][idx][0]:
                        idx = k
                        min_ci = self._taken_picture_list[n][idx][0]
                        
                # 今までの写真と多くは被っていない時
                if len(cover_list) == 0:
                    #範囲が多く被っていなくて、self._num_picture回未満写真を撮っていたらそのまま保存
                    if len(self._taken_picture_list[n]) != self._num_picture:
                        self._taken_picture_list[n].append([ci[n], observations[n]["agent_position"], picture_range_map])
                        self._taken_picture[n].append(observations[n]["rgb"])
                        reward[n] += ci[n]
                            
                    #範囲が多く被っていなくて、self._num_picture回以上写真を撮っていたら
                    else:
                        # 今回の写真が保存してある写真の１つでもCIが高かったらCIが最小の保存写真と入れ替え
                        if idx != -1:
                            ci_pre = self._taken_picture_list[n][idx][0]
                            self._taken_picture_list[n][idx] = [ci[n], observations[n]["agent_position"], picture_range_map]
                            self._taken_picture[n][idx] = observations[n]["rgb"]   
                            reward[n] += (ci[n] - ci_pre)     
                        
                # 1つとでも多く被っていた時    
                else:
                    min_idx = -1
                    min_ci_k = 1000
                    # 多く被った写真のうち、ciが最小のものを計算
                    for k in range(len(cover_list)):
                        idx_k = cover_list[k]
                        if self._taken_picture_list[n][idx_k][0] < min_ci_k:
                            min_ci_k = self._taken_picture_list[n][idx_k][0]
                            min_idx = idx_k
                                
                    # 被った割合分小さくなったCIでも保存写真の中の最小のCIより大きかったら交換
                    if self._compareWithChangedCI(picture_range_map, pre_fog_of_war_map, cover_list, ci[n], min_ci_k, min_idx) == True:
                        self._taken_picture_list[n][min_idx] = [ci[n], observations[n]["agent_position"], picture_range_map]
                        self._taken_picture[n][min_idx] = observations[n]["rgb"]   
                        reward[n] += (ci[n] - min_ci_k)  
            
        reward = torch.tensor(
            reward, dtype=torch.float, device=current_episode_reward.device
        ).unsqueeze(1)
        exp_area = np.array(exp_area)
        exp_area = torch.tensor(
           exp_area, dtype=torch.float, device=current_episode_reward.device
        ).unsqueeze(1)
        

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.float,
            device=current_episode_reward.device,
        )
        
        # episode ended
        for n in range(len(observations)):
            if masks[n].item() == 0.0:
                for i in range(self._num_picture):
                    if i < len(self._taken_picture_list[n]):
                        self.take_picture_writer.write(str(self._taken_picture_list[n][i][0]))
                        self.picture_position_writer.write(str(self._taken_picture_list[n][i][1][0]) + "," + str(self._taken_picture_list[n][i][1][1]) + "," + str(self._taken_picture_list[n][i][1][2]))
                        self._ci += self._taken_picture_list[n][i][0]
                    else:
                        self.take_picture_writer.write(" ")
                        self.picture_position_writer.write(" ")
                        
                self.take_picture_writer.writeLine()
                self.picture_position_writer.writeLine()
                self._taken_picture[n] = []
                self._taken_picture_list[n] = []

        current_episode_reward += reward
        running_episode_stats["reward"] += (1 - masks) * current_episode_reward
        current_episode_exp_area = exp_area
        running_episode_stats["exp_area"] += (1 - masks) * current_episode_exp_area
        running_episode_stats["count"] += 1 - masks

        for k, v in self._extract_scalars_from_infos(infos).items():
            v = torch.tensor(
                v, dtype=torch.float, device=current_episode_reward.device
            ).unsqueeze(1)
            if k not in running_episode_stats:
                running_episode_stats[k] = torch.zeros_like(
                    running_episode_stats["count"]
                )

            running_episode_stats[k] += (1 - masks) * v

    
        current_episode_reward *= masks
        current_episode_exp_area *= masks

        if self._static_encoder:
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        rollouts.insert(
            batch,
            recurrent_hidden_states,
            actions,
            actions_log_probs,
            values,
            reward,
            masks,
        )

        pth_time += time.time() - t_update_stats

        return pth_time, env_time, self.envs.num_envs

    def _update_agent(self, ppo_cfg, rollouts):
        t_update_model = time.time()
        with torch.no_grad():
            last_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }
            next_value = self.actor_critic.get_value(
                last_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
            ).detach()

        rollouts.compute_returns(
            next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )

        value_loss, action_loss, dist_entropy = self.agent.update(rollouts)

        rollouts.after_update()

        return (
            time.time() - t_update_model,
            value_loss,
            action_loss,
            dist_entropy,
        )

    def train(self, log_manager, date) -> None:
        r"""Main method for training PPO.

        Returns:
            None
        """
        
        #ログ出力設定
        #time, reward
        reward_logger = log_manager.createLogWriter("reward")
        #time, learning_rate
        learning_rate_logger = log_manager.createLogWriter("learning_rate")
        #time, cnn_0, cnn_1, cnn_2, linear
        visual_encoder_logger = log_manager.createLogWriter("visual_encoder")
        #time, found, forward, left, right, look_up, look_down
        action_logger = log_manager.createLogWriter("action_prob")
        #time, picture, ci, episode_length
        metrics_logger = log_manager.createLogWriter("metrics")
        #time, losses_value, losses_policy
        loss_logger = log_manager.createLogWriter("loss")
        
        self.take_picture_writer = log_manager.createLogWriter("take_picture")
        self.picture_position_writer = log_manager.createLogWriter("picture_position")
        
        self._ci = 0.0
        self._take_picture = 0

        self.envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME)
        )
        
        for i in range(self.envs.num_envs):
            self._taken_picture.append([])
            self._taken_picture_list.append([])

        ppo_cfg = self.config.RL.PPO
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)
        self._setup_actor_critic_agent(ppo_cfg)
        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        rollouts = RolloutStorageOracle(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            self.envs.observation_spaces[0],
            self.envs.action_spaces[0],
            ppo_cfg.hidden_size,
        )
        rollouts.to(self.device)

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)

        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])

        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        batch = None
        observations = None

        current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        current_episode_exp_area = torch.zeros(self.envs.num_envs, 1)
        running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
            exp_area=torch.zeros(self.envs.num_envs, 1),
        )
        window_episode_stats = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size)
        )

        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0
        count_checkpoints = 0

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),
        )

        os.makedirs(self.config.TENSORBOARD_DIR + "/" + date, exist_ok=True)

        for update in range(self.config.NUM_UPDATES):
            if ppo_cfg.use_linear_lr_decay:
                lr_scheduler.step()

            if ppo_cfg.use_linear_clip_decay:
                self.agent.clip_param = ppo_cfg.clip_param * linear_decay(
                    update, self.config.NUM_UPDATES
                )

            for step in range(ppo_cfg.num_steps):
                if (step + update*ppo_cfg.num_steps) % 500 == 0:
                    print("STEP: " + str(step + update*ppo_cfg.num_steps))
                (
                    delta_pth_time,
                    delta_env_time,
                    delta_steps,
                ) = self._collect_rollout_step(
                    rollouts, current_episode_reward, current_episode_exp_area, running_episode_stats
                )
                pth_time += delta_pth_time
                env_time += delta_env_time
                count_steps += delta_steps

            (
                delta_pth_time,
                value_loss,
                action_loss,
                dist_entropy,
            ) = self._update_agent(ppo_cfg, rollouts)
            pth_time += delta_pth_time
                
            for k, v in running_episode_stats.items():
                window_episode_stats[k].append(v.clone())

            deltas = {
                k: (
                    (v[-1] - v[-2]).sum().item()
                    if len(v) > 1
                    else v[0].sum().item()
                )
                for k, v in window_episode_stats.items()
            }
            deltas["count"] = max(deltas["count"], 1.0)

            #csv
            reward_logger.writeLine(str(count_steps) + "," + str(deltas["reward"] / deltas["count"]))
            learning_rate_logger.writeLine(str(count_steps) + "," + str(lr_scheduler._last_lr[0]))

            total_actions = rollouts.actions.shape[0] * rollouts.actions.shape[1]
            total_found_actions = int(torch.sum(rollouts.actions == 0).cpu().numpy())
            total_forward_actions = int(torch.sum(rollouts.actions == 1).cpu().numpy())
            total_left_actions = int(torch.sum(rollouts.actions == 2).cpu().numpy())
            total_right_actions = int(torch.sum(rollouts.actions == 3).cpu().numpy())
            total_look_up_actions = int(torch.sum(rollouts.actions == 4).cpu().numpy())
            total_look_down_actions = int(torch.sum(rollouts.actions == 5).cpu().numpy())
            assert total_actions == (total_found_actions + total_forward_actions + 
                total_left_actions + total_right_actions + total_look_up_actions + 
                total_look_down_actions
            )
                
            # csv
            action_logger.writeLine(
                str(count_steps) + "," + str(total_found_actions/total_actions) + ","
                + str(total_forward_actions/total_actions) + "," + str(total_left_actions/total_actions) + ","
                + str(total_right_actions/total_actions) + "," + str(total_look_up_actions/total_actions) + ","
                + str(total_look_down_actions/total_actions)
            )
            metrics = {
                k: v / deltas["count"]
                for k, v in deltas.items()
                if k not in {"reward", "count"}
            }

            if len(metrics) > 0:                    
                logger.info("CI:" + str(self._ci/deltas["count"]) + " TAKE_PICTURE:" + str(self._take_picture / deltas["count"]))
                metrics_logger.writeLine(str(count_steps) + "," +str(self._take_picture / deltas["count"]) + "," + str(self._ci/deltas["count"]) + "," + str(metrics["exp_area"]) + "," + str(metrics["raw_metrics.agent_path_length"]))
                self._ci = 0.0
                self._take_picture = 0
                    
                logger.info(metrics)
            
            loss_logger.writeLine(str(count_steps) + "," + str(value_loss) + "," + str(action_loss))
                

            # log stats
            if update > 0 and update % self.config.LOG_INTERVAL == 0:
                logger.info(
                    "update: {}\tfps: {:.3f}\t".format(
                        update, count_steps / (time.time() - t_start)
                    )
                )

                logger.info(
                    "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                    "frames: {}".format(
                        update, env_time, pth_time, count_steps
                    )
                )

                logger.info(
                    "Average window size: {}  {}".format(
                        len(window_episode_stats["count"]),
                        "  ".join(
                            "{}: {:.3f}".format(k, v / deltas["count"])
                            for k, v in deltas.items()
                            if k != "count"
                        ),
                    )
                )

            # checkpoint model
            if update % self.config.CHECKPOINT_INTERVAL == 0:
                self.save_checkpoint(
                f"ckpt.{count_checkpoints}.pth", dict(step=count_steps)
                )
                count_checkpoints += 1

        self.envs.close()
            
            
    # 写真を撮った範囲のマップを作成
    def _create_picture_range_map(self, top_down_map, fog_of_war_map):
        # 0: 壁など, 1: 写真を撮った範囲, 2: 巡回可能領域
        picture_range_map = np.zeros_like(top_down_map)
        for i in range(len(top_down_map)):
            for j in range(len(top_down_map[0])):
                if top_down_map[i][j] != 0:
                    if fog_of_war_map[i][j] == 1:
                        picture_range_map[i][j] = 1
                    else:
                        picture_range_map[i][j] = 2
                        
        return picture_range_map
            
    # fog_mapがpre_fog_mapと閾値以上の割合で被っているか
    def _check_percentage_of_fog(self, fog_map, pre_fog_map, threshold=0.25):
        y = len(fog_map)
        x = len(fog_map[0])
        
        num = 0 #fog_mapのMAP_VALID_POINTの数
        num_covered = 0 #pre_fog_mapと被っているグリッド数
        
        for i in range(y):
            for j in range(x):
                # fog_mapで写真を撮っている範囲の時
                if fog_map[i][j] == 1:
                    num += 1
                    # fogとpre_fogがかぶっている時
                    if pre_fog_map[i][j] == 1:
                        num_covered += 1
                        
        if num == 0:
            per = 0.0
        else:
            per = num_covered / num
        
        if per < threshold:
            return False
        else:
            return True
        
    # fog_mapがidx以外のpre_fog_mapと被っている割合を算出
    def _cal_rate_of_fog_other(self, fog_map, pre_fog_of_war_map_list, cover_list, idx):
        y = len(fog_map)
        x = len(fog_map[0])
        
        num = 0.0 #fog_mapのMAP_VALID_POINTの数
        num_covered = 0.0 #pre_fog_mapのどれかと被っているグリッド数
        
        for i in range(y):
            for j in range(x):
                # fog_mapで写真を撮っている範囲の時
                if fog_map[i][j] == 1:
                    num += 1
                    
                    # 被っているmapを検査する
                    for k in range(len(cover_list)):
                        map_idx = cover_list[k]
                        if map_idx == idx:
                            continue
                        
                        pre_map = pre_fog_of_war_map_list[map_idx]
                        # fogとpre_fogがかぶっている時
                        if pre_map[i][j] == 1:
                            num_covered += 1
                            break
                        
        if num == 0:
            rate = 0.0
        else:
            rate = num_covered / num
        
        return rate
    
    
    def _compareWithChangedCI(self, picture_range_map, pre_fog_of_war_map_list, cover_list, ci, pre_ci, idx):
        rate = self._cal_rate_of_fog_other(picture_range_map, pre_fog_of_war_map_list, cover_list, idx)
        ci = ci * (1-rate) # k以外と被っている割合分小さくする
        if ci > pre_ci:
            return True
        else:
            return False
        

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        log_manager: LogManager,
        date: str,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        
        #ログ出力設定
        #time, reward
        eval_reward_logger = log_manager.createLogWriter("reward")
        #time, take_picture, ci, exp_area, path_length
        eval_metrics_logger = log_manager.createLogWriter("metrics")
        eval_take_picture_writer = log_manager.createLogWriter("take_picture")
        eval_picture_position_writer = log_manager.createLogWriter("picture_position")
        
        self._ci = 0.0
        self._take_picture = 0
        
        #フォルダがない場合は、作成
        p_dir = pathlib.Path("./log/" + date + "/val/Matrics")
        if not p_dir.exists():
            p_dir.mkdir(parents=True)
        
        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
        print("PATH")
        print(checkpoint_path)

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        ppo_cfg = config.RL.PPO

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()

        logger.info(f"env config: {config}")
        self.envs = construct_envs(config, get_env_class(config.ENV_NAME))
        self._setup_actor_critic_agent(ppo_cfg)

        self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic
        
        self._taken_picture = []
        self._taken_picture_list = []
        for i in range(self.envs.num_envs):
            self._taken_picture.append([])
            self._taken_picture_list.append([])

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )
        current_episode_exp_area = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )
        
        test_recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            self.config.NUM_PROCESSES,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device
        )
        stats_episodes = dict()  # dict of dicts that stores stats per episode
        raw_metrics_episodes = dict()

        rgb_frames = [
            [] for _ in range(self.config.NUM_PROCESSES)
        ]  # type: List[List[np.ndarray]]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR+"/"+date, exist_ok=True)

        pbar = tqdm.tqdm(total=self.config.TEST_EPISODE_COUNT)
        self.actor_critic.eval()
        while (
            len(stats_episodes) < self.config.TEST_EPISODE_COUNT
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                (
                    _,
                    actions,
                    _,
                    test_recurrent_hidden_states,
                ) = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )

                prev_actions.copy_(actions)

            outputs = self.envs.step([a[0].item() for a in actions])
 
            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            batch = batch_obs(observations, device=self.device)
            
            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )
            
            reward = []
            ci = []
            exp_area = []
            matrics = []
            fog_of_war_map = []
            top_down_map = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                reward.append(rewards[i][0][0])
                ci.append(rewards[i][0][1])
                exp_area.append(rewards[i][0][2])
                matrics.append(rewards[i][1])
                fog_of_war_map.append(infos[i]["picture_range_map"]["fog_of_war_mask"])
                top_down_map.append(infos[i]["picture_range_map"]["map"])
            
            for n in range(len(observations)):
                #TAKE_PICTUREが呼び出されたかを検証
                if ci[n] != -sys.float_info.max:
                    # 今回撮ったpicture(p_n)が保存してあるpicture(p_k)とかぶっているkを保存
                    cover_list = [] 
                    picture_range_map = self._create_picture_range_map(top_down_map[n], fog_of_war_map[n])
                    self._take_picture += 1
                    # p_kのそれぞれのpicture_range_mapのリスト
                    pre_fog_of_war_map = [sublist[2] for sublist in self._taken_picture_list[n]]
                    
                    # それぞれと閾値より被っているか計算
                    idx = -1
                    min_ci = ci[n]
                    for k in range(len(pre_fog_of_war_map)):
                        # 閾値よりも被っていたらcover_listにkを追加
                        if self._check_percentage_of_fog(picture_range_map, pre_fog_of_war_map[k]) == True:
                            cover_list.append(k)
                            
                        #ciの最小値の写真を探索(１つも被っていない時用)
                        if min_ci < self._taken_picture_list[n][idx][0]:
                            idx = k
                            min_ci = self._taken_picture_list[n][idx][0]
                        
                    # 今までの写真と多くは被っていない時
                    if len(cover_list) == 0:
                        #範囲が多く被っていなくて、self._num_picture回未満写真を撮っていたらそのまま保存
                        if len(self._taken_picture_list[n]) != self._num_picture:
                            self._taken_picture_list[n].append([ci[n], observations[n]["agent_position"], picture_range_map])
                            self._taken_picture[n].append(observations[n]["rgb"])
                            reward[n] += ci[n]
                            
                        #範囲が多く被っていなくて、self._num_picture回以上写真を撮っていたら
                        else:
                            # 今回の写真が保存してある写真の１つでもCIが高かったらCIが最小の保存写真と入れ替え
                            if idx != -1:
                                ci_pre = self._taken_picture_list[n][idx][0]
                                self._taken_picture_list[n][idx] = [ci[n], observations[n]["agent_position"], picture_range_map]
                                self._taken_picture[n][idx] = observations[n]["rgb"]   
                                reward[n] += (ci[n] - ci_pre)     
                        
                    # 1つとでも多く被っていた時    
                    else:
                        min_idx = -1
                        min_ci_k = 1000
                        # 多く被った写真のうち、ciが最小のものを計算
                        for k in range(len(cover_list)):
                            idx_k = cover_list[k]
                            if self._taken_picture_list[n][idx_k][0] < min_ci_k:
                                min_ci_k = self._taken_picture_list[n][idx_k][0]
                                min_idx = idx_k
                                
                        # 被った割合分小さくなったCIでも保存写真の中の最小のCIより大きかったら交換
                        if self._compareWithChangedCI(picture_range_map, pre_fog_of_war_map, cover_list, ci[n], min_ci_k, min_idx) == True:
                            self._taken_picture_list[n][min_idx] = [ci[n], observations[n]["agent_position"], picture_range_map]
                            self._taken_picture[n][min_idx] = observations[n]["rgb"]   
                            reward[n] += (ci[n] - min_ci_k)  
                
            
            reward = torch.tensor(
                reward, dtype=torch.float, device=self.device
            ).unsqueeze(1)
            exp_area = np.array(exp_area)
            exp_area = torch.tensor(
                exp_area, dtype=torch.float, device=self.device
            ).unsqueeze(1)

            current_episode_reward += reward
            current_episode_exp_area += exp_area
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []

            for i in range(n_envs):
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended
                if not_done_masks[i].item() == 0:
                    logger.info("Exp_area: " + str(current_episode_exp_area[i]))
                    eval_take_picture_writer.write(str(len(stats_episodes)) + "," + str(current_episodes[i].episode_id) + "," + str(n))
                    eval_picture_position_writer.write(str(len(stats_episodes)) + "," + str(current_episodes[i].episode_id) + "," + str(n))
                    for j in range(self._num_picture):
                        if j < len(self._taken_picture_list[i]):
                            eval_take_picture_writer.write(str(self._taken_picture_list[i][j][0]))
                            eval_picture_position_writer.write(str(self._taken_picture_list[i][j][1][0]) + "," + str(self._taken_picture_list[i][j][1][1]) + "," + str(self._taken_picture_list[i][j][1][2]))
                            self._ci += self._taken_picture_list[i][j][0]
                        else:
                            eval_take_picture_writer.write(" ")
                            eval_picture_position_writer.write(" ")
                        
                    eval_take_picture_writer.writeLine()
                    eval_picture_position_writer.writeLine()

                    next_episodes = self.envs.current_episodes()
                    envs_to_pause = []
                    
                    pbar.update()
                    episode_stats = dict()
                    episode_stats["reward"] = current_episode_reward[i].item()
                    episode_stats["exp_area"] = current_episode_exp_area[i].item()
                    episode_stats.update(
                        self._extract_scalars_from_info(infos[i])
                    )
                    current_episode_reward[i] = 0
                    current_episode_exp_area[i] = 0
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats
                    
                    raw_metrics_episodes[
                        current_episodes[i].scene_id + '.' + 
                        current_episodes[i].episode_id
                    ] = infos[i]["raw_metrics"]

                    if len(self.config.VIDEO_OPTION) > 0:
                        if len(rgb_frames[i]) == 0:
                            frame = observations_to_image(observations[i], infos[i], actions[i].cpu().numpy())
                            rgb_frames[i].append(frame)
                        picture = rgb_frames[i][-1]
                        for j in range(50):
                           rgb_frames[i].append(picture) 
                        metrics=self._extract_scalars_from_info(infos[i])
                        name_ci = 0.0
                        
                        for j in range(len(self._taken_picture_list[i])):
                            name_ci += self._taken_picture_list[i][j][0]
                        
                        name_ci = str(name_ci) + "-" + str(len(stats_episodes))
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR+"/"+date,
                            images=rgb_frames[i],
                            episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=metrics,
                            tb_writer=writer,
                            name_ci=name_ci,
                        )
        
                        # Save taken picture
                        metric_strs = []
                        in_metric = ['exp_area', 'ci']
                        for k, v in metrics.items():
                            if k in in_metric:
                                metric_strs.append(f"{k}={v:.2f}")
                        
                        #logger.info(infos[i])
                        name_p = 0.0  
                            
                        for j in range(len(self._taken_picture_list[i])):
                            eval_picture_top_logger = log_manager.createLogWriter("picture_top_" + str(len(stats_episodes)) + "_" + str(current_episodes[i].episode_id) + "_" + str(j) + "_" + str(n) + "_" + str(checkpoint_index))
                
                            for k in range(len(self._taken_picture_list[i][j][2])):
                                for l in range(len(self._taken_picture_list[i][j][2][0])):
                                    eval_picture_top_logger.write(str(self._taken_picture_list[i][j][2][k][l]))
                                eval_picture_top_logger.writeLine()
                                
                            name_p = self._taken_picture_list[i][j][0]
                            picture_name = "episode=" + str(len(stats_episodes))+ "-" + str(current_episodes[i].episode_id)+ "-ckpt=" + str(checkpoint_index) + "-" + str(j) + "-" + str(name_p)
                            dir_name = "./taken_picture/" + date 
                            if not os.path.exists(dir_name):
                                os.makedirs(dir_name)
                        
                            picture = self._taken_picture[i][j]
                            plt.figure()
                            ax = plt.subplot(1, 1, 1)
                            ax.axis("off")
                            plt.imshow(picture)
                            plt.subplots_adjust(left=0.1, right=0.95, bottom=0.05, top=0.95)
                            path = dir_name + "/" + picture_name + ".png"
                        
                            plt.savefig(path)
                            logger.info("Picture created: " + path)
                        
                        #Save score_matrics
                        if matrics[i] is not None:
                            eval_matrics_logger = log_manager.createLogWriter("Matrics/matrics_" + str(current_episodes[i].episode_id) + "_" + str(checkpoint_index))
                            for j in range(matrics[i].shape[0]):
                                for k in range(matrics[i].shape[1]):
                                    eval_matrics_logger.write(str(matrics[i][j][k]))
                                eval_matrics_logger.writeLine("")
                            
                            
                        rgb_frames[i] = []
                        
                    self._taken_picture[i] = []
                    self._taken_picture_list[i] = []

                # episode continues
                elif len(self.config.VIDEO_OPTION) > 0:
                    frame = observations_to_image(observations[i], infos[i], actions[i].cpu().numpy())
                    rgb_frames[i].append(frame)

            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        num_episodes = len(stats_episodes)
        
        aggregated_stats = dict()
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum([v[stat_key] for v in stats_episodes.values()])
                / num_episodes
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")
        


        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]
            
        """
        writer.add_scalar("eval/average_reward", aggregated_stats["reward"],
            step_id,
        )
        """
        
        eval_reward_logger.writeLine(str(step_id) + "," + str(aggregated_stats["reward"]))

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        """
        writer.add_scalar("eval/episode_length", metrics["episode_length"], step_id)
        writer.add_scalar("eval/picture", metrics["picture"], step_id)
        writer.add_scalar("eval/ci", metrics["ci"], step_id)
        """

        logger.info("CI:" + str(self._ci/num_episodes) + " TAKE_PICTURE:" + str(self._take_picture / num_episodes))
        eval_metrics_logger.writeLine(str(step_id) + "," +str(self._take_picture / num_episodes) + "," + str(self._ci/ num_episodes) + "," + str(metrics["exp_area"]) + "," + str(metrics["raw_metrics.agent_path_length"]))
        self._ci = 0.0
        self._take_picture = 0
        

        ##Dump metrics JSON
        if 'RAW_METRICS' in config.TASK_CONFIG.TASK.MEASUREMENTS:
            if not os.path.exists(config.TENSORBOARD_DIR_EVAL +'/metrics'):
                os.mkdir(config.TENSORBOARD_DIR_EVAL +'/metrics')
            with open(config.TENSORBOARD_DIR_EVAL + '/metrics/' + checkpoint_path.split('/')[-1] + '.json', 'w') as fp:
                json.dump(raw_metrics_episodes, fp)

        # if not os.path.exists(config.TENSORBOARD_DIR_EVAL +'/traj_metrics'):
        #     os.mkdir(config.TENSORBOARD_DIR_EVAL +'/traj_metrics')
        # with open(config.TENSORBOARD_DIR_EVAL +'/traj_metrics/' + checkpoint_path.split('/')[-1] + '.json', 'w') as fp:
        #     json.dump(traj_metrics_episodes, fp)



        self.envs.close()

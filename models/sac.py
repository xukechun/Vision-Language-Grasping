import numpy as np
import torch
import torch.nn.functional as F
import torch.distributions as td
from torch.optim import Adam
from models.networks import CLIPGraspFusion, QNetwork, Policy


class ViLG(object):
    def __init__(self, grasp_dim, args):

        self.device = args.device

        # state-action feature
        self.vilg_fusion = CLIPGraspFusion(grasp_dim, args.width, args.layers, args.heads, self.device).to(device=self.device)
        self.feature_optim = Adam(self.vilg_fusion.parameters(), lr=args.lr)

        # critic
        self.critic = QNetwork(args.width, args.hidden_size).to(device=self.device)            
        self.critic_target = QNetwork(args.width, args.hidden_size).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        # policy
        self.policy = Policy(args.width, args.hidden_size).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)


        if not args.evaluate:
            # SAC parameters
            self.gamma = args.gamma
            self.tau = args.tau
            self.alpha = args.alpha # self.alpha = 0
            self.target_update_interval = args.target_update_interval
            self.automatic_entropy_tuning = args.automatic_entropy_tuning

            if self.automatic_entropy_tuning:
                # self.log_alpha = torch.tensor(0., requires_grad=True)
                self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True)
                self.alpha = self.log_alpha.exp()
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)
            
            # hard update
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data)
            for k,v in self.vilg_fusion.named_parameters():
                if 'clip' in k:
                    v.requires_grad = False # fix parameters
                    # print(v.requires_grad)

            self.vilg_fusion.train()
            self.critic.train()
            self.critic_target.train()
            self.policy.train()

        else:
            self.vilg_fusion.eval()
            self.critic.eval()
            self.critic_target.eval()
            self.policy.eval()


    def get_fusion_feature(self, bboxes, pos_bboxes, text, actions):
        vilg_feature, clip_probs, vig_attn = self.vilg_fusion(bboxes, pos_bboxes, text, actions)
        return vilg_feature, clip_probs, vig_attn


    def select_action(self, bboxes, pos_bboxes, text, actions, evaluate=False):
        sa, clip_probs, vig_attn = self.get_fusion_feature(bboxes, pos_bboxes, text, actions)
        logits = self.policy(sa)
        if sa.shape[0] == 1:
            logits = logits.unsqueeze(0)
        if actions.shape[1] == 1:
            logits = logits.unsqueeze(0)
        mu = logits.argmax(-1)  # [B,]
        cate_dist = td.Categorical(logits=logits)
        pi = cate_dist.sample()  # [B,]

        action = pi if not evaluate else mu

        return logits.detach().cpu().numpy(), action.detach().cpu().numpy()[0], clip_probs.detach().cpu().numpy(), vig_attn.detach().cpu().numpy()[0]


    def forward(self, bboxes, pos_bboxes, text, actions):
        sa, probs, _ = self.get_fusion_feature(bboxes, pos_bboxes, text, actions)
        logits = self.policy(sa)
        
        if sa.shape[0] == 1:
            logits = logits.unsqueeze(0)
        if actions.shape[1] == 1:
            logits = logits.unsqueeze(0)

        cate_dist = td.Categorical(logits=logits)
        pi = cate_dist.sample()  # [B,]
        log_prob = cate_dist.log_prob(pi).unsqueeze(-1)
        
        qf1, qf2 = self.critic(sa)
        return log_prob, qf1, qf2
        

    def update_parameters(self, memory, batch_size, updates):

        # Sample a batch from memory
        lang_batch, bboxes_batch, pos_bboxes_batch, grasps_batch, action_batch, reward_batch, mask_batch, next_bboxes_batch, next_pos_bboxes_batch, next_grasps_batch = memory.sample(batch_size=batch_size)

        bboxes_batch = torch.FloatTensor(bboxes_batch).to(self.device)
        grasps_batch = torch.FloatTensor(grasps_batch).to(self.device)
        pos_bboxes_batch = torch.FloatTensor(pos_bboxes_batch).to(self.device)
        next_bboxes_batch = torch.FloatTensor(next_bboxes_batch).to(self.device)
        next_pos_bboxes_batch = torch.FloatTensor(next_pos_bboxes_batch).to(self.device)
        next_grasps_batch = torch.FloatTensor(next_grasps_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device)

        with torch.no_grad():
            next_sa, _, _ = self.get_fusion_feature(next_bboxes_batch, next_pos_bboxes_batch, lang_batch, next_grasps_batch)

            logits = self.policy(next_sa)
            if next_sa.shape[0] == 1:
                logits = logits.unsqueeze(0)
            if next_grasps_batch.shape[1] == 1:
                logits = logits.unsqueeze(0)

            logits_prob = F.softmax(logits, -1)
            z = logits_prob == 0.0
            z = z.float() * 1e-8
            next_log_probs = torch.log(logits_prob + z)

            qf1_next_target, qf2_next_target = self.critic_target(next_sa) # [B, A, 1]
            qf1_next_target = qf1_next_target.reshape(qf1_next_target.shape[0], -1) # [B, A]
            qf2_next_target = qf2_next_target.reshape(qf2_next_target.shape[0], -1) # [B, A]

            v1_target = (next_log_probs.exp() * (qf1_next_target - self.alpha * next_log_probs)).sum(-1, keepdim=True)
            v2_target = (next_log_probs.exp() * (qf2_next_target - self.alpha * next_log_probs)).sum(-1, keepdim=True)
            min_qf_next_target = torch.min(v1_target, v2_target)

            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
            
        sa, _, _ = self.get_fusion_feature(bboxes_batch, pos_bboxes_batch, lang_batch, grasps_batch)
        qf1, qf2 = self.critic(sa)  # Two Q-functions to mitigate positive bias in the policy improvement step        


        qf1 = qf1.squeeze(-1)
        qf2 = qf2.squeeze(-1)
        qf1 = qf1.gather(1, action_batch.to(torch.int64))
        qf2 = qf2.gather(1, action_batch.to(torch.int64))

        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = 0.5 * qf1_loss + 0.5 * qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.1)
        self.critic_optim.step()

        sa, probs, _ = self.get_fusion_feature(bboxes_batch, pos_bboxes_batch, lang_batch, grasps_batch)
        logits = self.policy(sa)
        if sa.shape[0] == 1:
            logits = logits.unsqueeze(0)
        if grasps_batch.shape[1] == 1:
            logits = logits.unsqueeze(0)

        logits_prob = F.softmax(logits, -1)
        z = logits_prob == 0.0
        z = z.float() * 1e-8
        log_probs = torch.log(logits_prob + z)
        entropy = -(log_probs.exp() * log_probs).sum(-1, keepdim=True)

        qf1_pi, qf2_pi = self.critic(sa)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        # Optional: add clip auxiliary loss
        # minimize the distance between clip probability distribution and grasp affordance distribution
        # if grasps_batch.shape[0] == 1:
        #     bbox_grasp_map = torch.zeros(pos_bboxes_batch.shape[1], grasps_batch.shape[1])
        #     dist_map = torch.zeros(pos_bboxes_batch.shape[1], grasps_batch.shape[1])
        #     for i in range(bbox_grasp_map.shape[0]):
        #         for j in range(bbox_grasp_map.shape[1]):
        #             dist = torch.norm(pos_bboxes_batch[0][i]-grasps_batch[0][j][:3])
        #             dist_map[i][j] = dist
        #             if dist < 0.05:
        #                 bbox_grasp_map[i][j] = 1
        # if grasps_batch.shape[0] == 1:
        #     pos_bboxes = pos_bboxes_batch[0].unsqueeze(1)
        #     pose_grasps = grasps_batch[0].unsqueeze(0)
        #     pos_grasps = pose_grasps[:, :, :3]
        #     dist_map = torch.norm((pos_bboxes-pos_grasps), dim=2)
        #     bbox_grasp_map = (dist_map<0.05).float()
        # clip_guided_logits = probs.float() @ bbox_grasp_map.to(self.device)

        # kl_loss = torch.nn.KLDivLoss(log_target=True, reduction="none")
        # input = F.log_softmax(logits, dim=1)
        # log_target = F.log_softmax(clip_guided_logits, dim=1)
        # clip_kl_loss = kl_loss(input, log_target).sum()

        if self.automatic_entropy_tuning:
            self.alpha = self.log_alpha.exp()
        
        # if updates < 300:
        #     policy_loss = -0.5 * ((min_qf_pi - self.alpha * log_probs) * log_probs.exp()).sum(-1).mean() + 0.5 * clip_kl_loss
        # else:
        policy_loss = -((min_qf_pi - self.alpha * log_probs) * log_probs.exp()).sum(-1).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.1)
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            self.alpha = self.log_alpha.exp()
            self.target_entropy = 0.98 * -np.log(1 / grasps_batch.shape[1])

            corr = (self.target_entropy - entropy).detach()
            alpha_loss = -(self.alpha * corr).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            
            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            # soft update
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        # update feature module with total loss
        with torch.no_grad():
            next_sa, probs, _ = self.get_fusion_feature(next_bboxes_batch, next_pos_bboxes_batch, lang_batch, next_grasps_batch)
            
            logits = self.policy(next_sa)
            if next_sa.shape[0] == 1:
                logits = logits.unsqueeze(0)
            if next_grasps_batch.shape[1] == 1:
                logits = logits.unsqueeze(0)

            next_log_probs = logits.log_softmax(-1)

            qf1_next_target, qf2_next_target = self.critic_target(next_sa) # [B, A, 1]

            qf1_next_target = qf1_next_target.reshape(qf1_next_target.shape[0], -1) # [B, A]
            qf2_next_target = qf2_next_target.reshape(qf2_next_target.shape[0], -1) # [B, A]

            v1_target = (next_log_probs.exp() * (qf1_next_target - self.alpha * next_log_probs)).sum(-1, keepdim=True)
            v2_target = (next_log_probs.exp() * (qf2_next_target - self.alpha * next_log_probs)).sum(-1, keepdim=True)
            min_qf_next_target = torch.min(v1_target, v2_target)

            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
            next_q_value = next_q_value[0]

        sa, probs, _ = self.get_fusion_feature(bboxes_batch, pos_bboxes_batch, lang_batch, grasps_batch)
        qf1, qf2 = self.critic(sa)  # Two Q-functions to mitigate positive bias in the policy improvement step        
        qf1 = torch.max(qf1.squeeze(-1), dim=1)[0]
        qf2 = torch.max(qf2.squeeze(-1), dim=1)[0]
        qf1_loss_ = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss_ = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        logits = self.policy(sa)
        if sa.shape[0] == 1:
            logits = logits.unsqueeze(0)
        if grasps_batch.shape[1] == 1:
            logits = logits.unsqueeze(0)

        log_probs = logits.log_softmax(-1)

        qf1_pi, qf2_pi = self.critic(sa)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        
        # Optional: add clip auxiliary loss
        # minimize the distance between clip probability distribution and grasp affordance distribution
        # if grasps_batch.shape[0] == 1:
        #     pos_bboxes = pos_bboxes_batch[0].unsqueeze(1)
        #     pose_grasps = grasps_batch[0].unsqueeze(0)
        #     pos_grasps = pose_grasps[:, :, :3]
        #     dist_map = torch.norm((pos_bboxes-pos_grasps), dim=2)
        #     bbox_grasp_map = (dist_map<0.05).float()
        # clip_guided_logits = probs.float() @ bbox_grasp_map.to(self.device)

        # # input should be a distribution in the log space
        # kl_loss = torch.nn.KLDivLoss(log_target=True, reduction="none")
        # input = F.log_softmax(logits, dim=1)
        # log_target = F.log_softmax(clip_guided_logits, dim=1)
        # clip_kl_loss_ = kl_loss(input, log_target).sum()

        if self.automatic_entropy_tuning:
            self.alpha = self.log_alpha.exp()

        policy_loss_ = -((min_qf_pi - self.alpha * log_probs) * log_probs.exp()).sum(-1).mean()

        # if updates < 300:
        #     total_loss = 0.33 * (0.5 * qf1_loss_ + 0.5 * qf2_loss_) + 0.33 * policy_loss_ + 0.33 * clip_kl_loss_
        # else:
        total_loss = 0.5 * (0.5 * qf1_loss_ + 0.5 * qf2_loss_) + 0.5 * policy_loss_

        self.feature_optim.zero_grad()
        total_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.vilg_fusion.parameters(), 0.1)
        self.feature_optim.step()
        
        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item(), total_loss.item()

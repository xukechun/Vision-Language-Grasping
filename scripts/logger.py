import os
import time
import datetime
import cv2
import torch
import numpy as np


class Logger:
    def __init__(self, case_dir=None, case=None):
        # Create directory to save data
        timestamp = time.time()
        timestamp_value = datetime.datetime.fromtimestamp(timestamp)
        if case is not None:
            name = case.split("/")[-1].split(".")[0] + "-"
            name = name[:-1]
        elif case_dir is not None:
            name = "test"
        else:
            name = "train"
        self.base_directory = os.path.join(
            os.path.abspath("logs"), timestamp_value.strftime("%Y-%m-%d-%H-%M-%S") + "-" + name,
        )
        print("Creating data logging session: %s" % (self.base_directory))
        self.color_heightmaps_directory = os.path.join(
            self.base_directory, "data", "color-heightmaps"
        )
        self.depth_heightmaps_directory = os.path.join(
            self.base_directory, "data", "depth-heightmaps"
        )
        self.bbox_heightmaps_directory = os.path.join(
            self.base_directory, "data", "bbox-heightmaps"
        )
        self.mask_directory = os.path.join(self.base_directory, "data", "masks")
        self.prediction_directory = os.path.join(self.base_directory, "data", "predictions")
        self.visualizations_directory = os.path.join(self.base_directory, "visualizations")
        self.transitions_directory = os.path.join(self.base_directory, "transitions")
        self.checkpoints_directory = os.path.join(self.base_directory, "checkpoints")

        self.reward_logs = []
        self.episode_reward_logs = []
        self.episode_step_logs = []
        self.episode_success_logs = []
        self.executed_action_logs = []

        if not os.path.exists(self.color_heightmaps_directory):
            os.makedirs(self.color_heightmaps_directory)
        if not os.path.exists(self.depth_heightmaps_directory):
            os.makedirs(self.depth_heightmaps_directory)
        if not os.path.exists(self.bbox_heightmaps_directory):
            os.makedirs(self.bbox_heightmaps_directory)
        if not os.path.exists(self.mask_directory):
            os.makedirs(self.mask_directory)
        if not os.path.exists(self.prediction_directory):
            os.makedirs(self.prediction_directory)
        if not os.path.exists(self.visualizations_directory):
            os.makedirs(self.visualizations_directory)
        if not os.path.exists(self.transitions_directory):
            os.makedirs(os.path.join(self.transitions_directory))
        if not os.path.exists(self.checkpoints_directory):
            os.makedirs(os.path.join(self.checkpoints_directory))

        if case is not None or case_dir is not None:
            self.result_directory = os.path.join(self.base_directory, "results")
            if not os.path.exists(self.result_directory):
                os.makedirs(self.result_directory)


    def save_heightmaps(self, iteration, color_heightmap, depth_heightmap):
        color_heightmap = cv2.cvtColor(color_heightmap, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            os.path.join(self.color_heightmaps_directory, "%06d.color.png" % (iteration)),
            color_heightmap,
        )
        depth_heightmap = np.round(depth_heightmap * 100000).astype(
            np.uint16
        )  # Save depth in 1e-5 meters
        cv2.imwrite(
            os.path.join(self.depth_heightmaps_directory, "%06d.depth.png" % (iteration)),
            depth_heightmap,
        )
    
    def save_bbox_images(self, iteration, bbox_images):
        for i in range(len(bbox_images)):
            bbox_image = cv2.cvtColor(bbox_images[i], cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                os.path.join(self.bbox_heightmaps_directory, "%06d.%d.bbox.png" % (iteration, i)),
                bbox_image,
            )

    def write_to_log(self, log_name, log):
        np.savetxt(
            os.path.join(self.transitions_directory, "%s.log.txt" % log_name), log, delimiter=" "
        )

    def save_predictions(self, iteration, pred, name="push"):
        cv2.imwrite(
            os.path.join(self.prediction_directory, "%06d.png" % (iteration)), pred,
        )

    def save_visualizations(self, iteration, affordance_vis, name):
        cv2.imwrite(
            os.path.join(self.visualizations_directory, "%06d.%s.png" % (iteration, name)),
            affordance_vis,
        )
    
    # Save model parameters
    def save_checkpoint(self, model, datatime, suffix="", ckpt_path=None):
        if ckpt_path is None:
            ckpt_path = "sac_checkpoint_{}_{}.pth".format(datatime, suffix)
            ckpt_path = os.path.join(self.base_directory, self.checkpoints_directory, ckpt_path)
        print('Saving models to {}'.format(ckpt_path))
        # torch.save(model.state_dict(), ckpt_path)
        torch.save({'feature_state_dict': model.vilg_fusion.state_dict(),
                    'policy_state_dict': model.policy.state_dict(),
                    'critic_state_dict': model.critic.state_dict(),
                    'critic_target_state_dict': model.critic_target.state_dict(),
                    'critic_optimizer_state_dict': model.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': model.policy_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, model, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)

            model.vilg_fusion.load_state_dict(checkpoint['feature_state_dict'])
            model.policy.load_state_dict(checkpoint['policy_state_dict'])
            model.critic.load_state_dict(checkpoint['critic_state_dict'])
            model.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            model.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            model.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            
            if evaluate:
                model.vilg_fusion.eval()
                model.policy.eval()
                model.critic.eval()
                model.critic_target.eval()
            else:
                model.vilg_fusion.train()
                model.policy.train()
                model.critic.train()
                model.critic_target.train()


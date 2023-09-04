import argparse
import numpy as np
import random
import datetime
import torch

import utils
from constants import WORKSPACE_LIMITS
from environment_sim import Environment
from logger import Logger
from grasp_detetor import Graspnet
from models.replay_memory import ReplayMemory
from models.sac import ViLG



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', action='store', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=1234, metavar='N',
                    help='random seed (default: 1234)')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='batch size (default: 1)')
    parser.add_argument('--evaluate', action='store', type=bool, default=False)
    parser.add_argument('--load_model', action='store', type=bool, default=False)
    parser.add_argument('--model_path', action='store', type=str, default='')
    parser.add_argument('--save_model_interval', type=int, default=500, metavar='N',
                        help='episode interval to save model')

    parser.add_argument('--num_obj', action='store', type=int, default=15)
    parser.add_argument('--num_episode', action='store', type=int, default=5000)
    parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy every 10 episode (default: True)')
    parser.add_argument('--max_episode_step', type=int, default=8)

    # Transformer paras
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--layers', type=int, default=1) # cross attention layer
    parser.add_argument('--heads', type=int, default=8)

    # SAC parameters
    # default parameters of SAC
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=512, metavar='N',
                        help='size of replay buffer (default: 512)')

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    
    # set device and seed
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # parameters
    num_obj = args.num_obj
    num_episode = args.num_episode

    # load environment
    env = Environment(gui=True)
    env.seed(args.seed)
    # env_sim = Environment(gui=False)
    # load logger
    logger = Logger()
    # load graspnet
    graspnet = Graspnet()
    # load vision-language-action model
    agent = ViLG(grasp_dim=7, args=args)
    if args.load_model:
        logger.load_checkpoint(agent, args.model_path, args.evaluate)

    # Memory
    memory = ReplayMemory(args.replay_size, args.seed)

    # training
    iteration = 0
    updates = 0

    for episode in range(num_episode):
        episode_reward = 0
        episode_steps = 0
        done = False
        reset = False
        episilo = min(0.6 * np.power(1.0002, episode), 0.99)

        while not reset:
            env.reset()
            # env_sim.reset()
            lang_goal = env.generate_lang_goal()
            if episode < 500:
                warmup_num_obj = 8
                reset = env.add_objects(warmup_num_obj, WORKSPACE_LIMITS)
            else:
                reset = env.add_objects(num_obj, WORKSPACE_LIMITS)
            print(f"\033[032m Reset environment of episode {episode}, language goal {lang_goal}\033[0m")

        while not done:
            # check if one of the target objects is in the workspace:
            out_of_workspace = []
            for obj_id in env.target_obj_ids:
                pos, _, _ = env.obj_info(obj_id)
                if pos[0] < WORKSPACE_LIMITS[0][0] or pos[0] > WORKSPACE_LIMITS[0][1] \
                    or pos[1] < WORKSPACE_LIMITS[1][0] or pos[1] > WORKSPACE_LIMITS[1][1]:
                    out_of_workspace.append(obj_id)
            if len(out_of_workspace) == len(env.target_obj_ids):
                print("\033[031m Target objects are not in the scene!\033[0m")
                break     

            if episode_steps == 0:
                color_image, depth_image, mask_image = utils.get_true_heightmap(env)
                bbox_images, bbox_positions = utils.get_true_bboxs(env, color_image, depth_image, mask_image)

                # graspnet
                pcd = utils.get_fuse_pointcloud(env)
                # Note that the object poses here can be replaced by the bbox 3D positions with identity rotations
                with torch.no_grad():
                    grasp_pose_set, _, _ = graspnet.grasp_detection(pcd, env.get_true_object_poses())
                print("Number of grasping poses", len(grasp_pose_set))
                if len(grasp_pose_set) == 0:
                    break
                # preprocess
                remain_bbox_images, bboxes, pos_bboxes, grasps = utils.preprocess(bbox_images, bbox_positions, grasp_pose_set, (args.patch_size, args.patch_size))
                if bboxes == None:
                    break

            if len(grasp_pose_set) == 1:
                action_idx = 0
            else:
                if np.random.randn () <= episilo: # greedy policy 
                    with torch.no_grad():
                        logits, action_idx, clip_probs, vig_attn = agent.select_action(bboxes, pos_bboxes, lang_goal, grasps)
                else:
                    action_idx = np.random.randint(0, len(grasp_pose_set))

            action = grasp_pose_set[action_idx]

            if len(memory) >= args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha, feature_loss = agent.update_parameters(memory, args.batch_size, updates)
                    updates += 1

            reward, done = env.step(action)
            if episode < 500:
                if reward > -1 and reward < 0:
                    reward = -1
            episode_steps += 1
            iteration += 1
            episode_reward += reward
            print("\033[034m Episode: {}, total numsteps: {}, reward: {}\033[0m".format(episode, iteration, round(reward, 2), done))

            # next state
            next_color_image, next_depth_image, next_mask_image = utils.get_true_heightmap(env)
            next_bbox_images, next_bbox_positions = utils.get_true_bboxs(env, next_color_image, next_depth_image, next_mask_image)
            next_pcd = utils.get_fuse_pointcloud(env)
            with torch.no_grad():
                next_grasp_pose_set, _, _ = graspnet.grasp_detection(next_pcd, env.get_true_object_poses())
            print("Number of grasping poses in next state", len(next_grasp_pose_set))
            if len(next_grasp_pose_set) == 0:
                break

            # preprocess
            next_remain_bbox_images, next_bboxes, next_pos_bboxes, next_grasps = utils.preprocess(next_bbox_images, next_bbox_positions, next_grasp_pose_set, (args.patch_size, args.patch_size))
            if next_bboxes == None:
                break

            # Ignore the "done" signal if it comes from hitting the max step horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == args.max_episode_step else float(not done)

            memory.push(bboxes.detach().cpu().numpy()[0], pos_bboxes.detach().cpu().numpy()[0], grasps.detach().cpu().numpy()[0], lang_goal, action_idx, reward, next_bboxes.detach().cpu().numpy()[0], next_pos_bboxes.detach().cpu().numpy()[0], next_grasps.detach().cpu().numpy()[0], mask) # Append transition to memory
            
            # record
            logger.save_heightmaps(iteration, color_image, depth_image)
            logger.save_bbox_images(iteration, remain_bbox_images)
            logger.reward_logs.append(reward)
            logger.executed_action_logs.append(action)
            logger.write_to_log('reward', logger.reward_logs)
            logger.write_to_log('executed_action', logger.executed_action_logs)
            
            if done or episode_steps == args.max_episode_step:
                break

            color_image = next_color_image
            depth_image = next_depth_image
            mask_image = next_mask_image
            remain_bbox_images = next_remain_bbox_images
            bboxes = next_bboxes
            pos_bboxes = next_pos_bboxes
            grasps = next_grasps
            grasp_pose_set = next_grasp_pose_set

        
        if (episode + 1) % args.save_model_interval == 0:
            logger.save_checkpoint(agent, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), str(episode))
        logger.episode_reward_logs.append(episode_reward)
        logger.episode_step_logs.append(episode_steps)
        logger.episode_success_logs.append(done)
        logger.write_to_log('episode_reward', logger.episode_reward_logs)
        logger.write_to_log('episode_step', logger.episode_step_logs)
        logger.write_to_log('episode_success', logger.episode_success_logs)
        print("\033[034m Episode: {}, total numsteps: {}, episode steps: {}, episode reward: {}, success: {}\033[0m".format(episode, iteration, episode_steps, round(episode_reward, 2), done))
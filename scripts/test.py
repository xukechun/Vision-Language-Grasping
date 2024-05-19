import argparse
import numpy as np
import random
import torch
import os

import utils
from env.constants import WORKSPACE_LIMITS
from env.environment_sim import Environment
from logger import Logger
from grasp_detetor import Graspnet
from models.vilg_sac import ViLG



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', action='store', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=1234, metavar='N',
                    help='random seed (default: 1234)')
    parser.add_argument('--evaluate', dest='evaluate', action='store_false', default=True)
    parser.add_argument('--testing_case_dir', action='store', type=str, default='testing_cases/')
    parser.add_argument('--testing_case', action='store', type=str, default=None)

    parser.add_argument('--load_model', dest='load_model', action='store_true', default=False)
    parser.add_argument('--model_path', action='store', type=str, default='')

    parser.add_argument('--num_episode', action='store', type=int, default=15)
    parser.add_argument('--max_episode_step', type=int, default=8)

    # Transformer paras
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--layers', type=int, default=1) # cross attention layer
    parser.add_argument('--heads', type=int, default=8)

    # SAC parameters
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')

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
    num_episode = args.num_episode

    # load environment
    env = Environment(gui=True)
    env.seed(args.seed)
    # load logger
    logger = Logger(case_dir=args.testing_case_dir)
    # load graspnet
    graspnet = Graspnet()
    # load vision-language-action model
    agent = ViLG(grasp_dim=7, args=args)
    if args.load_model:
        logger.load_checkpoint(agent, args.model_path, args.evaluate)
        
    if os.path.exists(args.testing_case_dir):
        filelist = os.listdir(args.testing_case_dir)
        filelist.sort(key=lambda x:int(x[4:6]))
    if args.testing_case != None:
        filelist = [args.testing_case]
    case = 0
    iteration = 0
    for f in filelist:
        f = os.path.join(args.testing_case_dir, f)

        logger.episode_reward_logs = []
        logger.episode_step_logs = []
        logger.episode_success_logs = []
        for episode in range(num_episode):
            episode_reward = 0
            episode_steps = 0
            done = False
            reset = False

            while not reset:
                env.reset()
                reset, lang_goal = env.add_object_push_from_file(f)
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
                # remain_bbox_images, bboxes, grasps = utils.preprocess(bbox_images, grasp_pose_set, args.patch_size)
                remain_bbox_images, bboxes, pos_bboxes, grasps = utils.preprocess(bbox_images, bbox_positions, grasp_pose_set, (args.patch_size, args.patch_size))
                logger.save_bbox_images(iteration, remain_bbox_images)
                logger.save_heightmaps(iteration, color_image, depth_image)
                if bboxes == None:
                    break

                if len(grasp_pose_set) == 1:
                    action_idx = 0
                else:
                    with torch.no_grad():
                        logits, action_idx, clip_probs, vig_attn = agent.select_action(bboxes, pos_bboxes, lang_goal, grasps, evaluate=args.evaluate)

                action = grasp_pose_set[action_idx]
                reward, done = env.step(action)
                iteration += 1
                episode_steps += 1
                episode_reward += reward
                print("\033[034m Episode: {}, step: {}, reward: {}\033[0m".format(episode, episode_steps, round(reward, 2)))

                if episode_steps == args.max_episode_step:
                    break

            
            logger.episode_reward_logs.append(episode_reward)
            logger.episode_step_logs.append(episode_steps)
            logger.episode_success_logs.append(done)
            logger.write_to_log('episode_reward', logger.episode_reward_logs)
            logger.write_to_log('episode_step', logger.episode_step_logs)
            logger.write_to_log('episode_success', logger.episode_success_logs)
            print("\033[034m Episode: {}, episode steps: {}, episode reward: {}, success: {}\033[0m".format(episode, episode_steps, round(episode_reward, 2), done))

            if episode == num_episode - 1:
                avg_success = sum(logger.episode_success_logs)/len(logger.episode_success_logs)
                avg_reward = sum(logger.episode_reward_logs)/len(logger.episode_reward_logs)
                avg_step = sum(logger.episode_step_logs)/len(logger.episode_step_logs)
                
                success_steps = []
                for i in range(len(logger.episode_success_logs)):
                    if logger.episode_success_logs[i]:
                        success_steps.append(logger.episode_step_logs[i])
                if len(success_steps) > 0:
                    avg_success_step = sum(success_steps) / len(success_steps)
                else:
                    avg_success_step = 1000

                result_file = os.path.join(logger.result_directory, "case" + str(case) + ".txt")
                with open(result_file, "w") as out_file:
                    out_file.write(
                        "%s %.18e %.18e %.18e %.18e\n"
                        % (
                            lang_goal,
                            avg_success,
                            avg_step,
                            avg_success_step,
                            avg_reward,
                        )
                    )
                case += 1
                print("\033[034m Language goal: {}, average steps: {}/{}, average reward: {}, average success: {}\033[0m".format(lang_goal, avg_step, avg_success_step, avg_reward, avg_success))
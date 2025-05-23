'''export a gif video from a trajectory file'''

import torch
from argparse import ArgumentParser

from aime_nob.data import NPZTrajectory, HDF5Trajectory
from aime_nob.utils import save_gif

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--traj', type=str, required=True)
    parser.add_argument('--output_gif', type=str, default='debug.gif')
    parser.add_argument('--image_key', type=str, default='image')
    parser.add_argument('--fps', type=float, default=25)
    args = parser.parse_args()

    if args.traj.endswith('npz'):
        traj = NPZTrajectory(args.traj)
    else:
        traj = HDF5Trajectory(args.traj)

    data = traj.get_trajectory()

    video = data[args.image_key]
    save_gif(args.output_gif, video, args.fps)
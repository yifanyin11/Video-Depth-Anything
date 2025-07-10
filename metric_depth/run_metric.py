# Copyright (2025) Bytedance Ltd. and/or its affiliates 

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
import argparse
import numpy as np
import os
import torch
import glob
from PIL import Image
from tqdm import tqdm

from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames, save_video

def read_image_frames(folder_path, max_len=-1, max_res=1280):
    """Read frames from a folder containing image files."""
    image_files = sorted(glob.glob(os.path.join(folder_path, '*.jpg')) + 
                         glob.glob(os.path.join(folder_path, '*.png')))
    
    if max_len > 0:
        image_files = image_files[:max_len]
    
    frames = []
    for img_path in tqdm(image_files, desc=f"Reading frames from {os.path.basename(folder_path)}"):
        img = Image.open(img_path).convert('RGB')
        
        # Resize if needed
        if max_res > 0:
            w, h = img.size
            scale = min(max_res / w, max_res / h)
            if scale < 1:
                new_w, new_h = int(w * scale), int(h * scale)
                img = img.resize((new_w, new_h), Image.LANCZOS)
        
        frames.append(np.array(img))
    
    return frames, len(frames) / 30.0  # Assuming 30 fps as default

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Depth Anything')
    parser.add_argument('--input_video', type=str, default='../assets/davis_rollercoaster.mp4')
    parser.add_argument('--root_dir', type=str, default=None,
                        help='Root directory containing video folders with frame images')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--max_res', type=int, default=1280)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vitl'])
    parser.add_argument('--max_len', type=int, default=-1, help='maximum length of the input video, -1 means no limit')
    parser.add_argument('--target_fps', type=int, default=-1, help='target fps of the input video, -1 means the original fps')
    parser.add_argument('--fp32', action='store_true', help='model infer with torch.float32, default is torch.float16')
    parser.add_argument('--save_npz', action='store_true', help='save depths as npz')
    parser.add_argument('--save_exr', action='store_true', help='save depths as exr files')
    parser.add_argument('--save_viz', action='store_true', help='save visualization files')
    parser.add_argument('--grayscale', action='store_true', help='do not apply colorful palette')
    parser.add_argument('--folders_file', type=str, default=None, 
                        help='Path to a text file containing video folder paths to process')
    
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    video_depth_anything = VideoDepthAnything(**model_configs[args.encoder])
    video_depth_anything.load_state_dict(torch.load(f'./checkpoints/metric_video_depth_anything_{args.encoder}.pth', map_location='cpu'), strict=True)
    video_depth_anything = video_depth_anything.to(DEVICE).eval()

    # Process dataset directory structure if provided
    if args.root_dir is not None:
        # Determine video folders to process
        if args.folders_file is not None:
            # Read folder paths from the specified text file
            with open(args.folders_file, 'r') as f:
                video_folders = [line.strip() for line in f.readlines() if line.strip()]
            print(f"Loaded {len(video_folders)} video folders from {args.folders_file}")
        else:
            # Use the original method to find all subdirectories
            video_folders = [f for f in glob.glob(os.path.join(args.root_dir, '*')) if os.path.isdir(f)]
            print(f"Found {len(video_folders)} video folders to process")
        
        for video_folder in tqdm(video_folders, desc="Processing video folders"):
            video_name = os.path.basename(video_folder)
            print(f"\nProcessing video folder: {video_name}")
            
            # Read frames from image files
            frames, estimated_fps = read_image_frames(video_folder, args.max_len, args.max_res)
            target_fps = args.target_fps if args.target_fps > 0 else estimated_fps
            
            if len(frames) == 0:
                print(f"No frames found in {video_folder}, skipping.")
                continue
                
            # Convert frames list to numpy array
            frames_array = np.array(frames)
                
            # Process frames
            depths, fps = video_depth_anything.infer_video_depth(
                frames_array, target_fps, input_size=args.input_size, device=DEVICE, fp32=args.fp32)
            
            # Save outputs in the video folder
            processed_video_path = os.path.join(video_folder, f'{video_name}_src.mp4')
            depth_vis_path = os.path.join(video_folder, f'{video_name}_vis.mp4')
            
            if args.save_viz:
                save_video(frames_array, processed_video_path, fps=fps)
                save_video(depths, depth_vis_path, fps=fps, is_depths=True, grayscale=args.grayscale)
            
            if args.save_npz:
                depth_npz_path = os.path.join(video_folder, f'{video_name}_depths.npz')
                np.savez_compressed(depth_npz_path, depths=depths)
                
            if args.save_exr:
                depth_exr_dir = os.path.join(video_folder, f'{video_name}_depths_exr')
                os.makedirs(depth_exr_dir, exist_ok=True)
                import OpenEXR
                import Imath
                for i, depth in enumerate(depths):
                    output_exr = f"{depth_exr_dir}/frame_{i:05d}.exr"
                    header = OpenEXR.Header(depth.shape[1], depth.shape[0])
                    header["channels"] = {
                        "Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
                    }
                    exr_file = OpenEXR.OutputFile(output_exr, header)
                    exr_file.writePixels({"Z": depth.tobytes()})
                    exr_file.close()
            
    # Original code path for processing a single video file
    else:
        frames, target_fps = read_video_frames(args.input_video, args.max_len, args.target_fps, args.max_res)
        
        # Convert frames to numpy array if it's a list
        if isinstance(frames, list):
            frames = np.array(frames)
            
        depths, fps = video_depth_anything.infer_video_depth(frames, target_fps, input_size=args.input_size, device=DEVICE, fp32=args.fp32)
        
        video_name = os.path.basename(args.input_video)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        processed_video_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_src.mp4')
        depth_vis_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_vis.mp4')
        save_video(frames, processed_video_path, fps=fps)
        save_video(depths, depth_vis_path, fps=fps, is_depths=True, grayscale=args.grayscale)

        if args.save_npz:
            depth_npz_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_depths.npz')
            np.savez_compressed(depth_npz_path, depths=depths)
            
        if args.save_exr:
            depth_exr_dir = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_depths_exr')
            os.makedirs(depth_exr_dir, exist_ok=True)
            import OpenEXR
            import Imath
            for i, depth in enumerate(depths):
                output_exr = f"{depth_exr_dir}/frame_{i:05d}.exr"
                header = OpenEXR.Header(depth.shape[1], depth.shape[0])
                header["channels"] = {
                    "Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
                }
                exr_file = OpenEXR.OutputFile(output_exr, header)
                exr_file.writePixels({"Z": depth.tobytes()})
                exr_file.close()





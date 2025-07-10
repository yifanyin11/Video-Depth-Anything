import os
import glob
import argparse
import numpy as np
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Divide video folders into groups for parallel processing')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory containing video folders')
    parser.add_argument('--num_groups', type=int, default=8, help='Number of groups to divide into (default: 8)')
    parser.add_argument('--output_dir', type=str, default='./gpu_splits', help='Directory to save the group files')
    parser.add_argument('--create_scripts', action='store_true', help='Create shell scripts for processing (default: False)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all video folders in the root directory
    video_folders = [f for f in glob.glob(os.path.join(args.root_dir, '*')) if os.path.isdir(f)]
    
    if not video_folders:
        print(f"No video folders found in {args.root_dir}")
        return
    
    print(f"Found {len(video_folders)} video folders in {args.root_dir}")
    
    # Shuffle the folders to ensure even distribution of different sized videos
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(video_folders)
    
    # Divide folders into groups
    num_folders = len(video_folders)
    folders_per_group = num_folders // args.num_groups
    remainder = num_folders % args.num_groups
    
    start_idx = 0
    
    # Create a group file for each GPU
    for i in range(args.num_groups):
        # Add an extra folder to some groups if there's a remainder
        extra = 1 if i < remainder else 0
        end_idx = start_idx + folders_per_group + extra
        
        # Get the folders for this group
        group_folders = video_folders[start_idx:end_idx]
        
        # Write folder paths to a text file
        output_file = os.path.join(args.output_dir, f"group_{i}.txt")
        with open(output_file, 'w') as f:
            for folder in group_folders:
                f.write(f"{folder}\n")
        
        print(f"Group {i}: {len(group_folders)} folders, saved to {output_file}")
        start_idx = end_idx
    
    # Only create processing scripts if the flag is set
    if args.create_scripts:
        create_processing_scripts(args.num_groups, args.output_dir)
    
def create_processing_scripts(num_groups, output_dir):
    """Create shell scripts to process each group on a separate GPU"""
    
    script_template = """#!/bin/bash
# Process group {group_id} on GPU {group_id}
export CUDA_VISIBLE_DEVICES={group_id}

# Read folder paths from group file
while IFS= read -r folder
do
    echo "Processing folder: $folder"
    # python run_inference.py --root_dir "$folder" --save_npz
    # Uncomment the line below if you want to run metric depth as well
    python metric_depth/run_metric.py --root_dir "$folder" --save_npz
done < "{group_file}"
"""
    
    for i in range(num_groups):
        script_path = os.path.join(output_dir, f"process_group_{i}.sh")
        group_file = os.path.join(output_dir, f"group_{i}.txt")
        
        with open(script_path, 'w') as f:
            f.write(script_template.format(group_id=i, group_file=group_file))
        
        # Make the script executable
        os.chmod(script_path, 0o755)
        
    print(f"Created {num_groups} processing scripts in {output_dir}")
    print("Run all scripts in parallel with:")
    print(f"for i in {{0..{num_groups-1}}}; do {output_dir}/process_group_$i.sh & done")

if __name__ == "__main__":
    main()

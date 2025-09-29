#!/usr/bin/env python3
"""
Script to run testcases for ALL SWE-bench-verified tasks with their corresponding patches.

This script:
1. Loads trajectory files to get all available tasks
2. Processes ALL trajectories in the specified type
3. For each trajectory, applies its patch and runs testcases
4. Saves results with test information added back to trajectory data

Usage:
    # Run all trajectories
    python run_manual_patch_test.py --traj_type cheating --output_file results.json
    
    # Run a subset (first 10)
    python run_manual_patch_test.py --traj_type cheating --max_trajectories 10 --output_file results.json
    
    # List available trajectories
    python run_manual_patch_test.py --traj_type cheating --list_trajectories
"""

import os
import json
import argparse
import tempfile
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

# Import R2E-Gym modules
from src.r2egym.agenthub.environment.env import EnvArgs, RepoEnv
from src.r2egym.agenthub.agent.agent import AgentArgs
from src.r2egym.agenthub.trajectory.swebench_utils import swebench_report, swebench_parse
from src.r2egym.logging import setup_logging, INFO


# Available trajectory files to load from



def load_all_trajectories(file_path: str) -> Dict[str, Any]:
    # with open(TRAJECTORY_FILES[0], "r") as f:
    #     return json.load(f)
    if file_path.endswith(".json"):
        with open(file_path, "r") as f:
            return json.load(f)
    else:# load by jsonl
        trajectories = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    trajectories.append(json.loads(line))
        return trajectories



def get_instance_from_trajectory_id(traj_id: int, trajectories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Get instance data from trajectory ID."""
    if traj_id < 0 or traj_id >= len(trajectories):
        raise ValueError(f"Trajectory ID {traj_id} out of range [0, {len(trajectories)-1}]")
    
    return trajectories[traj_id]




def run_patch_test(ds: Dict[str, Any], patch: str, backend: str = "docker", 
                   max_reward_calc_time: int = 1200) -> Dict[str, Any]:
    """
    Run testcases for a specific SWE-bench task with a manually provided patch.
    
    Args:
        ds: The dataset entry containing task information
        patch: The patch content as a string
        backend: Backend to use ("docker" or "kubernetes")
        max_reward_calc_time: Timeout for test execution
        
    Returns:
        Dictionary containing test results and score information
    """
    instance_id = ds['instance_id']
    
    # Setup logging
    logger = setup_logging(
        name=f"manual_patch_test_{instance_id}",
        log_file=f"run_logs/manual_patch_test_{instance_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        console=True,
        level=INFO,
    )
    
    logger.info(f"Starting manual patch test for instance: {instance_id}")
    logger.info(f"Backend: {backend}")
    logger.info(f"Patch length: {len(patch)} characters")
    
    try:
        logger.info(f"Using trajectory data for instance: {ds['instance_id']}")
        logger.info(f"Docker image: {ds['docker_image']}")
        logger.info(f"Repository: {ds.get('repo', 'Unknown')}")
        
        # Create environment
        logger.info("Creating environment...")
        env_args = EnvArgs(ds=ds)
        env = RepoEnv(env_args, logger=logger, backend=backend)
        
        # Load agent configuration for commands
        agent_args = AgentArgs.from_yaml(
            Path("./src/r2egym/agenthub/config/r2egym/edit_fn_calling.yaml")
        )
        env.add_commands(agent_args.command_files)
        
        # Apply the manual patch
        logger.info("Applying manual patch...")
        # FIXME: load the gt_patch then test the reward
        # patch = env.runtime.commit.get_patch(test_file=False, non_test_file=True)
        apply_output, apply_error = env.runtime.apply_patch(patch)
        if apply_error != "0":
            logger.error(f"Patch application failed: {apply_output}")
            return {
                "success": False,
                "error": f"Patch application failed: {apply_output}",
                "instance_id": instance_id,
                "patch_applied": False
            }
        logger.info(f"Patch applied successfully: {patch}")
        
        # Calculate reward (run tests)
        logger.info("Running testcases...")
        start_time = time.time()
        reward, test_output = env.runtime._calculate_reward(get_test_output=True, timeout=max_reward_calc_time)
        test_time = time.time() - start_time
        
        logger.info(f"Test execution completed in {test_time:.2f} seconds")
        logger.info(f"Reward: {reward}")
        
        # Parse SWE-bench results
        logger.info("Parsing SWE-bench results...")
        try:
            swe_report = swebench_report(ds, test_output)
            swe_parse = swebench_parse(ds, test_output)
            logger.info(f"SWE-bench report: {swe_report}")
        except Exception as e:
            logger.warning(f"Failed to parse SWE-bench results: {e}")
            swe_report = {}
            swe_parse = {}
        
        
        # Close environment
        env.close()
        
        # Prepare results
        results = {
            "success": True,
            "instance_id": instance_id,
            "docker_image": ds["docker_image"],
            "patch_applied": True,
            "reward": reward,
            "test_time": test_time,
            "test_output": test_output,
            "swe_report": swe_report,
            "swe_parse": swe_parse,
        }
        
        logger.info(f"Manual patch test completed successfully")
        logger.info(f"Final score: {reward}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during manual patch test: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "error": str(e),
            "instance_id": instance_id,
            "patch_applied": False
        }


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(description="Run testcases for ALL SWE-bench tasks with their patches")
    parser.add_argument("--traj_type", type=str, required=True, choices=["cheating", "no_cheating"],
                        help="Type of trajectory to load")
    parser.add_argument("--backend", type=str, default="docker", choices=["docker", "kubernetes"],
                        help="Backend to use")
    parser.add_argument("--timeout", type=int, default=1200,
                        help="Timeout for test execution in seconds")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to save results JSON file")
    parser.add_argument("--list_trajectories", action="store_true",
                        help="List available trajectories and exit")
    parser.add_argument("--max_trajectories", type=int, default=None,
                        help="Maximum number of trajectories to process")
    parser.add_argument("--start_from", type=int, default=0,
                        help="Start processing from trajectory ID (for resuming)")
    parser.add_argument("--file_path", type=str, default='/home/ftl3370/R2E-Gym/traj/claude_trajs_all_50_filtered.json',
                        help="Path to load trajectories from")
    args = parser.parse_args()
    
    # Load all trajectories
    print("Loading trajectory files...")
    all_trajs = load_all_trajectories(args.file_path)
    trajectories = all_trajs[args.traj_type]
    # trajectories = all_trajs
    if not trajectories:
        print("Error: No trajectories found in the specified files")
        return 1
    
    total_trajectories = len(trajectories)
    print(f"Found {total_trajectories} trajectories total")
    
    # List trajectories if requested
    if args.list_trajectories:
        print("\nAvailable trajectories:")
        print("=" * 80)
        for i, traj in enumerate(trajectories[:20]):  # Show first 20
            traj_ds = traj['ds']
            print(f"ID {i:2d}: {traj_ds['instance_id']:30} | Reward: {traj['reward']}")
        if len(trajectories) > 20:
            print(f"... and {len(trajectories) - 20} more")
        print(f"\nTotal: {total_trajectories} trajectories")
        return 0
    
    # Determine range of trajectories to process
    start_idx = args.start_from
    if args.max_trajectories:
        end_idx = min(start_idx + args.max_trajectories, total_trajectories)
    else:
        end_idx = total_trajectories
    
    print(f"Processing trajectories {start_idx} to {end_idx-1} (total: {end_idx - start_idx})")
    print(f"Backend: {args.backend}")
    print(f"Timeout: {args.timeout}s")
    print(f"Output file: {args.output_file}")
    print("=" * 80)
    
    # Statistics tracking
    successful_tests = 0
    failed_tests = 0
    total_time = 0
    
    # Process all trajectories in the specified range
    for traj_id in range(start_idx, end_idx):
        traj = trajectories[traj_id]
        patch = traj['filtered_patch']
        # patch = traj['ds']['patch']
        instance_id = traj['ds']['instance_id']
        
        print(f"\n[{traj_id+1}/{end_idx}] Processing trajectory {traj_id}")
        print(f"Instance ID: {instance_id}")
        print(f"Original Reward: {traj['reward']}")
        print(f"Patch length: {len(patch)} characters")
        
        # Run the test for this trajectory
        start_time = time.time()
        results = run_patch_test(
            ds=traj['ds'],
            patch=patch,
            backend=args.backend,
            max_reward_calc_time=args.timeout
        )
        test_time = time.time() - start_time
        total_time += test_time
        
        # Update statistics
        if results['success']:
            successful_tests += 1
            score_change = results['reward'] - (traj['reward'] if isinstance(traj['reward'], (int, float)) else 0)
            print(f"✅ SUCCESS | Score: {results['reward']} (change: {score_change:+.1f}) | Time: {test_time:.1f}s")
        else:
            failed_tests += 1
            print(f"❌ FAILED | Error: {results['error']} | Time: {test_time:.1f}s")
        
        # Add test results back to the trajectory
        all_trajs[args.traj_type][traj_id]['patch_test_results'] = results
        
        # Save progress periodically (every 10 trajectories) and at the end
        if (traj_id + 1) % 10 == 0 or traj_id == end_idx - 1:
            print(f"💾 Saving progress to {args.output_file}...")
            with open(args.output_file, 'w') as f:
                json.dump(all_trajs, f, indent=2)
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Total trajectories processed: {end_idx - start_idx}")
    print(f"Successful tests: {successful_tests}")
    print(f"Failed tests: {failed_tests}")
    print(f"Success rate: {successful_tests/(end_idx-start_idx)*100:.1f}%")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average time per test: {total_time/(end_idx-start_idx):.1f} seconds")
    print(f"Results saved to: {args.output_file}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

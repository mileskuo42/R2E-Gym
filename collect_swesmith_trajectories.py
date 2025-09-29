#!/usr/bin/env python3
"""
Script for collecting trajectories on swesmith dataset with filtering logic.
Excludes rows already processed in hubert233/R2E-Smith and checks for existing results.
Usage: python collect_swesmith_trajectories.py [--max_workers N] [--start_idx N] [--k N]
"""

import os
import json
import time
import concurrent.futures
import threading
import argparse
from pathlib import Path
from typing import Set, List, Dict, Any, Optional
from datasets import load_dataset
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Import R2E-Gym modules
from src.r2egym.agenthub.run.edit import runagent
from src.r2egym.agenthub.trajectory.trajectory import Trajectory
from src.r2egym.logging import setup_logging, INFO

# Initialize logger
logger = setup_logging(
    name="swesmith_collector",
    log_file="run_logs/swesmith/collector.log",
    console=True,
    level=INFO,
)

# Initialize file lock for thread-safe writing
file_lock = threading.Lock()

def load_existing_instance_ids_from_r2e_smith(exclude_r2e_subset: bool = True) -> Set[str]:
    """
    Load instance_ids from hubert233/R2E-Smith dataset where source is 'in-swesmith-clean'
    
    Args:
        exclude_r2e_subset: If True, exclude trajectories with source 'in-swesmith-clean' (default: True)

    Returns:
        Set of instance_ids to exclude
    """
    if not exclude_r2e_subset:
        logger.info("Skipping exclusion of R2E-subset trajectories from R2E-Smith dataset")
        return set()

    logger.info("Loading hubert233/R2E-Smith dataset to get exclusion list...")
    try:
        r2e_smith_ds = load_dataset("hubert233/R2E-Smith")
        exclude_instance_ids = set()
        
        # Check all splits for instances with source 'in-swesmith-clean'
        for split_name, split_data in r2e_smith_ds.items():
            logger.info(f"Checking {split_name} split with {len(split_data)} entries")
            for row in split_data:
                if row.get('source') == 'in-swesmith-clean':
                    if 'instance_id' in row:
                        exclude_instance_ids.add(row['instance_id'])
        
        logger.info(f"Found {len(exclude_instance_ids)} instance_ids to exclude from R2E-Smith dataset")
        return exclude_instance_ids
        
    except Exception as e:
        logger.error(f"Error loading hubert233/R2E-Smith dataset: {e}")
        logger.info("Continuing without exclusions from R2E-Smith...")
        return set()

def load_existing_results(traj_dir: str, exp_name: str) -> Set[str]:
    """
    Load already processed instance_ids from existing trajectory results
    
    Args:
        traj_dir: Directory containing trajectory files
        exp_name: Experiment name for the trajectory file
        
    Returns:
        Set of instance_ids that are already processed
    """
    traj_dir_path = Path(traj_dir)
    jsonl_file = traj_dir_path / f"{exp_name}.jsonl"
    
    existing_instance_ids = set()
    
    if jsonl_file.exists():
        logger.info(f"Loading existing results from {jsonl_file}")
        try:
            with open(jsonl_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        trajectory_data = json.loads(line.strip())
                        if 'ds' in trajectory_data and 'instance_id' in trajectory_data['ds']:
                            existing_instance_ids.add(trajectory_data['ds']['instance_id'])
                    except json.JSONDecodeError as e:
                        logger.warning(f"Error parsing line {line_num} in {jsonl_file}: {e}")
                        continue
                        
            logger.info(f"Found {len(existing_instance_ids)} already processed instances")
        except Exception as e:
            logger.error(f"Error reading existing results: {e}")
    else:
        logger.info(f"No existing results file found at {jsonl_file}")
    
    return existing_instance_ids

def filter_dataset(ds_swesmith: List[Dict], 
                  exclude_from_r2e_smith: Set[str], 
                  exclude_existing: Set[str]) -> List[Dict]:
    """
    Filter the swesmith dataset to remove already processed instances
    
    Args:
        ds_swesmith: The swesmith dataset entries
        exclude_from_r2e_smith: Instance IDs to exclude from R2E-Smith
        exclude_existing: Instance IDs already in trajectory results
        
    Returns:
        Filtered dataset entries
    """
    logger.info(f"Starting with {len(ds_swesmith)} total entries in swesmith dataset")
    
    # Filter out entries that are in R2E-Smith
    filtered_r2e_smith = [
        entry for entry in ds_swesmith 
        if entry.get('instance_id') not in exclude_from_r2e_smith
    ]
    logger.info(f"After excluding R2E-Smith entries: {len(filtered_r2e_smith)} entries remain")
    
    # Filter out entries that are already processed
    filtered_final = [
        entry for entry in filtered_r2e_smith 
        if entry.get('instance_id') not in exclude_existing
    ]
    logger.info(f"After excluding existing results: {len(filtered_final)} entries remain")
    
    return filtered_final

def collect_swesmith_trajectories(
    traj_dir: str = "./traj",
    exp_name: str = "swesmith",
    max_workers: Optional[int] = None,
    max_steps: int = 100,
    max_steps_absolute: int = 100,
    llm_name: str = 'openai/qwen3-coder-plus-2025-07-22',
    use_fn_calling: bool = False,
    temperature: float = 0.7,
    max_tokens: int = 65536,
    backend: str = "docker",
    max_reward_calc_time: int = 300,  
    max_iterations: int = 1,
    scaffold: str = "r2egym",
    num_restarts: int = 1,
    args_path: Optional[str] = None,
    start_idx: int = 0,
    k: Optional[int] = None,  # If None, process all available entries
    exclude_r2e_subset: bool = True,
):
    """
    Main function to collect trajectories for swesmith dataset
    
    Args:
        traj_dir: Directory to save trajectories
        exp_name: Experiment name (default: "swesmith")
        max_workers: Maximum number of parallel workers
        max_steps: Maximum steps for agent execution
        max_steps_absolute: Maximum absolute steps
        llm_name: LLM model to use
        use_fn_calling: Whether to use function calling
        temperature: Temperature for LLM
        max_tokens: Maximum tokens per request
        backend: Backend to use ("docker" instead of "kubernetes")
        max_reward_calc_time: Timeout for reward calculation
        max_iterations: Maximum iterations
        scaffold: Scaffold type
        num_restarts: Number of restarts
        args_path: Path to custom agent args (if any)
        start_idx: Starting index for processing
        k: Number of entries to process (if None, process all)
        exclude_r2e_subset: Whether to exclude R2E-subset trajectories from R2E-Smith (default: True)
    """
    
    logger.info("Starting swesmith trajectory collection...")
    
    # Create trajectory directory
    traj_dir_path = Path(traj_dir)
    traj_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create logs directory
    Path(f"run_logs/{exp_name}").mkdir(parents=True, exist_ok=True)
    
    # Load swesmith dataset
    logger.info("Loading r2e-edits/swesmith-clean dataset...")
    try:
        ds_swesmith = load_dataset("r2e-edits/swesmith-clean")
        ds_swesmith = ds_swesmith['train']  # Use train split
        logger.info(f"Loaded {len(ds_swesmith)} entries from swesmith dataset")
    except Exception as e:
        logger.error(f"Error loading swesmith dataset: {e}")
        return
    
    # Load exclusion lists
    exclude_from_r2e_smith = load_existing_instance_ids_from_r2e_smith(exclude_r2e_subset)
    exclude_existing = load_existing_results(traj_dir, exp_name)
    
    # Convert dataset to list for filtering
    ds_swesmith_list = [ds_swesmith[i] for i in range(len(ds_swesmith))]
    
    # Filter dataset
    filtered_entries = filter_dataset(ds_swesmith_list, exclude_from_r2e_smith, exclude_existing)
    
    if not filtered_entries:
        logger.info("No entries to process after filtering!")
        return
    
    # Apply start_idx and k filtering
    if start_idx > 0:
        filtered_entries = filtered_entries[start_idx:]
        logger.info(f"After applying start_idx {start_idx}: {len(filtered_entries)} entries")
    
    if k is not None and k > 0:
        filtered_entries = filtered_entries[:k]
        logger.info(f"After applying k={k}: {len(filtered_entries)} entries")
    
    logger.info(f"Final entries to process: {len(filtered_entries)}")
    
    # Prepare JSONL file
    jsonl_file = traj_dir_path / f"{exp_name}.jsonl"
    
    # Process entries - single-threaded if max_workers is 1, otherwise parallel
    completed_count = 0
    failed_count = 0
    
    if max_workers == 1:
        logger.info("Running in single-threaded mode for debugging...")
        
        with open(jsonl_file, "a") as f:
            for i, entry in enumerate(filtered_entries):
                instance_id = entry.get('instance_id', f'unknown_{i}')
                logger.info(f"Processing {i+1}/{len(filtered_entries)}: {instance_id}")
                
                try:
                    result = runagent(
                        ds=entry,
                        exp_name=exp_name,
                        max_steps=max_steps,
                        num_restarts=num_restarts,
                        max_steps_absolute=max_steps_absolute,
                        llm_name=llm_name,
                        temperature=temperature,
                        use_fn_calling=use_fn_calling,
                        backend=backend,
                        max_reward_calc_time=max_reward_calc_time,
                        max_iterations=max_iterations,
                        scaffold=scaffold,
                        max_tokens=max_tokens,
                        args_path=args_path,
                    )
                    
                    if result is not None:
                        f.write(result + "\n")
                        f.flush()  # Ensure immediate write
                        completed_count += 1
                        logger.info(f"Completed {completed_count}/{len(filtered_entries)}: {instance_id}")
                    else:
                        failed_count += 1
                        logger.error(f"Failed to get result for instance: {instance_id}")
                        
                except Exception as e:
                    failed_count += 1
                    logger.error(f"Exception processing instance {instance_id}: {e}")
                    import traceback
                    logger.error(f"Full traceback: {traceback.format_exc()}")
    else:
        logger.info(f"Starting parallel processing with {max_workers} workers...")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_instance = {
                executor.submit(
                    runagent,
                    ds=entry,
                    exp_name=exp_name,
                    max_steps=max_steps,
                    num_restarts=num_restarts,
                    max_steps_absolute=max_steps_absolute,
                    llm_name=llm_name,
                    temperature=temperature,
                    use_fn_calling=use_fn_calling,
                    backend=backend,
                    max_reward_calc_time=max_reward_calc_time,
                    max_iterations=max_iterations,
                    scaffold=scaffold,
                    max_tokens=max_tokens,
                    args_path=args_path,
                ): entry.get('instance_id', f'unknown_{i}')
                for i, entry in enumerate(filtered_entries)
            }
            
            # Process completed futures
            with open(jsonl_file, "a") as f:
                for future in concurrent.futures.as_completed(future_to_instance):
                    instance_id = future_to_instance[future]
                    try:
                        result = future.result()
                        if result is not None:
                            with file_lock:
                                f.write(result + "\n")
                                f.flush()  # Ensure immediate write
                            completed_count += 1
                            logger.info(f"Completed {completed_count}/{len(filtered_entries)}: {instance_id}")
                        else:
                            failed_count += 1
                            logger.error(f"Failed to get result for instance: {instance_id}")
                    except Exception as e:
                        failed_count += 1
                        logger.error(f"Exception processing instance {instance_id}: {e}")
    
    logger.info(f"Trajectory collection completed!")
    logger.info(f"Successfully processed: {completed_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info(f"Results saved to: {jsonl_file}")

def main():
    """
    Main entry point with command-line argument parsing
    """
    parser = argparse.ArgumentParser(description="Collect swesmith trajectories")
    parser.add_argument("--max_workers", type=int, default=10, help="Maximum number of parallel workers")
    parser.add_argument("--start_idx", type=int, default=0, help="Starting index")
    parser.add_argument("--k", type=int, default=None, help="Number of entries to process (if None, process all)")
    parser.add_argument("--traj_dir", type=str, default="./traj", help="Trajectory directory")
    parser.add_argument("--exp_name", type=str, default="swesmith", help="Experiment name")
    parser.add_argument("--no_exclude_r2e_subset", type=bool, default=False, help="Don't exclude R2E-subset trajectories from R2E-Smith dataset")

    args = parser.parse_args()
    
    # Configuration based on test_smith.py
    config = {
        'traj_dir': args.traj_dir,
        'exp_name': args.exp_name,
        'max_workers': args.max_workers,
        'max_steps': 100,
        'max_steps_absolute': 100,
        # 'llm_name': 'openai/Qwen/Qwen3-235B-A22B-Instruct-2507',
        'llm_name': 'openai/glm-4.5',
        # 'llm_name': 'anthropic/claude-4-sonnet-20250514',
        'use_fn_calling': False,
        'temperature': 0.7,
        'max_tokens': 65536,
        'backend': "docker",  # Changed from kubernetes to docker
        'max_reward_calc_time': 300,
        'max_iterations': 1,
        'scaffold': "r2egym",
        'num_restarts': 1,
        # 'args_path': "src/r2egym/agenthub/config/r2egym/edit_non_fn_calling_qwen3coder.yaml",
        'args_path': "src/r2egym/agenthub/config/r2egym/edit_non_fn_calling_claude4s.yaml",
        'args_path': "src/r2egym/agenthub/config/r2egym/edit_non_fn_calling_glm45.yaml",
        'start_idx': args.start_idx,
        'k': args.k,
        'exclude_r2e_subset': not args.no_exclude_r2e_subset,
    }
    
    logger.info("Starting swesmith trajectory collection with configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Start trajectory collection
    collect_swesmith_trajectories(**config)

if __name__ == "__main__":
    main()
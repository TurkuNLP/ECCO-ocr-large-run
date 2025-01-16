import glob
import os
import re
import subprocess
import sys
import argparse

def count_lines(filelist):
    S=set()
    for fname in filelist:
        with open(fname) as f:
            for line in f:
                S.add(line.strip())
    return len(S)

def partitions_in_queue():
    S=set()
    #I cannot grasp why this wouldn't run!
    #result=subprocess.run("/usr/bin/squeue -u ginter",capture_output=True,text=True,shell=True)
    for line in sys.stdin:#result.stderr.split("\n"):
        match=re.match(".*\sEB([0-9]{1,3})\s",line)
        if match:
           S.add(int(match.group(1)))
    return S

def gather_completed_and_failed(partition):
    completed=count_lines(glob.glob(f"ECCO-BIG-RUN-OUT/rank_{partition}_of*.completed.txt"))
    failed=count_lines(glob.glob(f"ECCO-BIG-RUN-OUT/rank_{partition}_of*.failed.txt"))
    return completed,failed

def schedule_jobs(args):
    in_queue=partitions_in_queue()
    can_schedule=args.max_in_queue-len(in_queue)
    total=207614
    total_completed=0
    total_failed=0
    for partition in range(0,200):
        completed,failed=gather_completed_and_failed(partition)
        total_completed+=completed
        total_failed+=failed
        if partition in in_queue:
            print(f"# EB{partition:03} in queue or running completed:{completed} failed:{failed}")
            continue
        if completed+failed>=total//200:
            print(f"# EB{partition:03} DONE completed:{completed} failed:{failed}")
            continue
        print(f"# EB{partition:03} completed:{completed} failed:{failed}")
        if can_schedule>0:
            print(f"sbatch -J EB{partition:03} -o STDOUTERR-ECCO-BIG-RUN/{args.run_name}_{partition:03}.eo run_vllm_lumi.sh --worker-rank {partition}")
            can_schedule-=1
        else:
            print("# cannot schedule more this time")
    print(f"#TOTALS: completed: {total_completed}  failed: {failed}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-in-queue",
        type=int,
        default=200,
        help="Maximum number of jobs in queue"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="Name of the run for logfiles"
    )
    return parser.parse_args()
    
if __name__=="__main__":
    args=parse_args()
    schedule_jobs(args)


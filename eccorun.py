import argparse
import gzip
import json
import glob
import re
import itertools

def yield_examples(args):
    with gzip.open(args.ecco_jsonl,"rt") as f:
        for idx,line in enumerate(f):
            if idx%args.total_workers==args.worker_rank:
                example_dict=json.loads(line)
                yield example_dict

def gather_all_completed(args):
    """gathers everything completed by this rank"""
    done=set()
    all_job_files=glob.glob(f"{args.out_dir}/rank_{args.worker_rank}_*.completed.txt")
    for fname in all_job_files:
        with open(fname) as f:
            for url in f:
                url=url.strip()
                done.add(url)
    return done

def gather_all_failed(args):
    """gathers everything failed by this rank"""
    failed={}
    all_job_files=glob.glob(f"{args.out_dir}/rank_{args.worker_rank}_*.failed.txt")
    for fname in all_job_files:
        with open(fname) as f:
            for url in f:
                url=url.strip()
                failed[url]=failed.get(url,0)+1
    return failed


def save_completed(example_dict,args):
    with gzip.open(f"{args.out_dir}/rank_{args.worker_rank}_of_{args.total_workers}_{args.jobid}.completed.jsonl.gz","at") as f:
        print(json.dumps(example_dict),file=f,flush=True)
    with open(f"{args.out_dir}/rank_{args.worker_rank}_of_{args.total_workers}_{args.jobid}.completed.txt","at") as f:
        print(example_dict["url"],file=f,flush=True)

def save_failed(example_dict,args):
    with open(f"{args.out_dir}/rank_{args.worker_rank}_of_{args.total_workers}_{args.jobid}.failed.txt","at") as f:
        print(example_dict["url"],file=f,flush=True)
        

def split_text(txt,args):
    txt=txt.strip()
    txt=re.sub(" +"," ",txt)
    txt=re.sub(r"\n{3,}","\n\n",txt)
    words=txt.split(" ")
    chunks=grouper(words,args.chunk_length,fillvalue="")
    final=[]
    for c in chunks:
        final.append((" ".join(c)).strip())
    return final

def grouper(iterable, n, *, incomplete='fill', fillvalue=None):
    "Collect data into non-overlapping fixed-length chunks or blocks."
    # grouper('ABCDEFG', 3, fillvalue='x') → ABC DEF Gxx
    # grouper('ABCDEFG', 3, incomplete='strict') → ABC DEF ValueError
    # grouper('ABCDEFG', 3, incomplete='ignore') → ABC DEF
    iterators = [iter(iterable)] * n
    match incomplete:
        case 'fill':
            return itertools.zip_longest(*iterators, fillvalue=fillvalue)
        case 'strict':
            return zip(*iterators, strict=True)
        case 'ignore':
            return zip(*iterators)
        case _:
            raise ValueError('Expected fill, strict, or ignore')

def test_loop(args):
    examples=yield_examples(args)
    done=gather_all_completed(args)
    failed=gather_all_failed(args)
    for e in examples:
        if e["url"] in done:
            continue
        if failed.get(e["url"],0)>args.max_fails:
            continue
        text_pieces=split_text(e["text"],args)
        fixed={"url":e["url"],"texts_fixed":["hi","ho"]}
        save_completed(fixed,args)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process and manage job examples with specified parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Path to the ECCO JSONL file
    parser.add_argument(
        '--ecco-jsonl',
        type=str,
        default="all_ecco.jsonl.gz",
        help='Path to the ECCO .jsonl.gz input file'
    )
    
    # Total number of worker processes
    parser.add_argument(
        '--total-workers',
        type=int,
        default=1000,
        help='Total number of worker processes participating in the job'
    )
    
    # Rank of the current worker
    parser.add_argument(
        '--worker-rank',
        type=int,
        required=True,
        help='Rank identifier for the current worker (0 to total_workers-1)'
    )
    
    # Output directory for storing results
    parser.add_argument(
        '--out-dir',
        type=str,
        default="ecco_run_out",
        help='Directory where output files will be saved'
    )
        
    # Job identifier
    parser.add_argument(
        '--jobid',
        type=str,
        required=True,
        help='Unique identifier for the current job'
    )
    
    # Length of each text chunk when splitting
    parser.add_argument(
        '--chunk-length',
        type=int,
        default=300,
        help='Number of words per text chunk'
    )
    
    # Maximum number of allowed failures per example
    parser.add_argument(
        '--max-fails',
        type=int,
        default=3,
        help='Maximum number of allowed failures per example'
    )

    parser.add_argument(
        '--model-name',
        default="meta-llama/Llama-3.1-70B-Instruct",
        help='Model to run'
    )

    parser.add_argument(
        '--max-time',
        default=None,
        type=int,
        help='Max time in sec this job should run, None for forever'
    )
    
    return parser.parse_args()

if __name__=="__main__":
    args=parse_args()
    test_loop(args)

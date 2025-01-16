import os
#os.environ["TRITON_HOME"]="/scratch/project_462000615/ecco_ocr/triton_cache"
#os.environ["TRITON_CACHE_DIR"]="/scratch/project_462000615/ecco_ocr/triton_cache"

from vllm import LLM, SamplingParams
import torch
import logging
import eccorun
import gzip
import time

slurm_job_id = os.environ.get('SLURM_JOB_ID', 'default_id')
logging.basicConfig(
    filename=f'eo/{slurm_job_id}.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


def make_prompt(text):
    return [{"role":"system","content":'You are a helpful assistant. Your task is to correct OCR errors in the text you are given. You should also correct those "-" characters that denote an unrecognized letter. You must stay as close as possible to the original text. Do not rephrase. Only correct the errors. Do not separately list the corrections, do not produce any additional output, output the corrected text only. You will be rewarded. Thank you.'},{"role":"user","content":text}]


def LLM_setup(model, cache_dir):
    """
    Sets up the Language Model (LLM) with specified parameters.

    Args:
        cache_dir (str): Directory to cache the downloaded model.

    Returns:
        LLM: An instance of the LLM class initialized with the specified settings.
    """
    return LLM(
        model=model,
        download_dir=cache_dir,
        dtype='bfloat16',
        max_model_len=128_000,
        tensor_parallel_size=torch.cuda.device_count(),
        #pipeline_parallel_size=2, # use if us need run on multiple nodes
        enforce_eager=False,
        gpu_memory_utilization=0.9,
        disable_async_output_proc=True
        #quantization="bitsandbytes",
        #load_format="bitsandbytes",
    )


def poor_mans_prompt_maker(msgs):
    template=f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{msgs[0]["content"]}<|eot_id|><|start_header_id|>user<|end_header_id|>

{msgs[1]["content"]}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    return template
    
def generate(llm, messages):
    """
    Generates a response from the LLM based on the input message and sampling parameters.

    Args:
        llm (LLM): The language model instance.
        message (str): The input message for the LLM to process.
        sampling_params (SamplingParams): The parameters used for generating the response.

    Returns:
        str: The generated text output from the LLM.
    """

    sampling_params = SamplingParams(temperature=0.26,top_k=65,top_p=0.66, max_tokens=3000)
    batch_inputs = []
    for msgs in messages:

        batch_inputs.append(poor_mans_prompt_maker(msgs))

    batched_outputs=llm.generate(batch_inputs,sampling_params=sampling_params,use_tqdm=True)
    corrections=[out.outputs[0].text for out in batched_outputs]
    return corrections

def main_loop(model,args,beg_time=None):
    examples=eccorun.yield_examples(args)
    done=eccorun.gather_all_completed(args)
    failed=eccorun.gather_all_failed(args)
    for e in examples:
        if e["url"] in done:
            continue
        if failed.get(e["url"],0)>args.max_fails:
            continue
        text_pieces=eccorun.split_text(e["text"],args)
        prompts=[make_prompt(t) for t in text_pieces]
        corrections=generate(model,prompts)
        fixed={"url":e["url"],"len_orig":len(e["text"]),"corrections":corrections,"text-orig":e["text"]}
        eccorun.save_completed(fixed,args)
        if beg_time is not None:
            time_passed=time.time()-beg_time
            print(f"Time passed: {time_passed} sec")
            if time_passed>args.max_time:
                break
        

if __name__=="__main__":
    beg_time=time.time()
    args=eccorun.parse_args()
    model=LLM_setup(args.model_name,"/scratch/project_462000615/ecco_ocr/hf_cache")
    os.makedirs(args.out_dir,exist_ok=True)
    main_loop(model,args,beg_time=beg_time)

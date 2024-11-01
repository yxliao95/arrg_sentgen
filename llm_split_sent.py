import json
import logging
import os
import time
import argparse

import transformers
import torch

PROJ_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
file_name = os.path.basename(__file__)
log_dir = os.path.join(PROJ_ROOT_DIR, "outputs", "logs")
os.makedirs(log_dir, exist_ok=True)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S"))

LOGGER = logging.getLogger("root")
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(console_handler)


def llm_process(generator, prompt, sent_str_list):
    # About how to form the input message: https://huggingface.co/docs/transformers/main/en/main_classes/pipelines#transformers.TextGenerationPipeline.__call__
    
    batch_chats = []
    for sent_str in sent_str_list:
        chat_msg = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": sent_str},
        ]
        batch_chats.append(chat_msg)
    
    # default no truncation
    outputs = generator(
        batch_chats,
        batch_size=len(batch_chats),
        max_new_tokens=128,
        return_full_text=False,
        temperature=0.1, # Set low temperature for less randomness
        # top_p=1, # Use top-p sampling for more focused generation
    )
    
    # The output would be like this if return_full_text=False: 
    # [
    #     [{'generated_text': '[Decreased bibasilar parenchymal opacities.]\n[The bibasilar parenchymal opacities are now minimal.]'}], 
    #     [{'generated_text': '[STABLE SMALL LEFT PLEURAL EFFUSION.]'}], 
    #     [{'generated_text': '[FEEDING TUBE AND STERNAL PLATES ARE SEEN.]'}]
    # ]
    batch_sent_splits = []
    for out in outputs:
        asst_content = out[-1]["generated_text"]
        # remove "[" and "]" from the each split sentence
        sent_splits = [marked_sent.strip().lstrip("[").rstrip("]") for marked_sent in asst_content.split("\n")]
        batch_sent_splits.append(sent_splits)
        
    # LOGGER.debug(f"origianl sent || {sent_str_list}")
    # LOGGER.debug(f'output || {batch_sent_splits}')

    return batch_sent_splits
    
def load_model(model_id):
    """ will return a TextGenerationPipeline """
    generator = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    generator.tokenizer.pad_token_id = generator.model.config.eos_token_id[0]
    LOGGER.info("generator.model.config %s", generator.model.config)
    return generator

def load_prompt(prompt_path):
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = "".join(f.readlines())
    return prompt

def load_data(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        docs = [json.loads(line.strip()) for line in f.readlines()]
    return docs

def partition_list(lst, curr_partition, max_partition):
    # Split the dataset
    sublist_size = len(lst) // max_partition
    # 计算当前分区的起始和结束索引
    start = (curr_partition - 1) * sublist_size
    end = start + sublist_size if curr_partition != max_partition else len(lst)
    return lst[start:end]

def doc_sent_generator(docs):
    for proc_idx, doc in enumerate(docs):
        sentences = doc["sentences"]  # [[token, token, ...], ...]
        if len(sentences) == 0:
            yield doc["doc_key"], None, ""
        else:
            for sent_idx, tokens in enumerate(sentences):
                sent_str = " ".join(tokens)
                yield doc["doc_key"], sent_idx, sent_str

def write_jsonline_to(f, doc_key, sent_idx, sent_splits, original_sent):
    out_json = {"doc_key": doc_key, "sent_idx": sent_idx, "original_sent": original_sent, "sent_splits": sent_splits}
    out_json_str = json.dumps(out_json, separators=(',', ':')) # compact
    f.write(out_json_str + "\n")
    
def read_last_line(data_path):
    with open(data_path, "rb") as f:
        f.seek(-2, 2)  # 从文件末尾的倒数第二个字节开始
        while f.read(1) != b"\n":  # 查找换行符
            f.seek(-2, 1)
        last_line = f.readline().decode().strip()
    return last_line

def count_file_lines(file_path):
    with open(file_path, "rb") as f:
        return sum(1 for _ in f)
    
def check_memory():
    # 获取当前 GPU 设备的属性
    device = torch.cuda.current_device()
    device_properties = torch.cuda.get_device_properties(device)
    # 获取 GPU 总显存
    total_memory = device_properties.total_memory / 1024**3  # 转换为 GB
    # 获取Torch总占用显存
    total_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
    LOGGER.info(f"Memory reserved: {total_reserved:.2f} / {total_memory:.2f} GB")
    
def seconds_to_time_str(seconds):
    hours = seconds // 3600  # 1小时 = 3600秒
    minutes = (seconds % 3600) // 60  # 除去小时部分后，再计算分钟
    seconds = seconds % 60  # 剩余的秒数

    return f"{hours:.0f}h {minutes:.0f}min {seconds:.1f}s"
    
    
def get_args(max_partition):
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_bash", action="store_true")
    parser.add_argument("--partition", type=int, help=f'Split the dataset into three partitions, specify the current partition for this script to process, value should be in [1...{max_partition}]')
    return parser.parse_args()
    
if __name__ == "__main__":
    first_start = time.time()
    
    max_partition = 3
    curr_partition = 1 # The processing partition of the dataset
    args = get_args(max_partition)
    if args.from_bash:
        curr_partition = args.partition
        assert 1 <= curr_partition <= max_partition, f"--partition should be chosen in [1...{max_partition}]"
        
        file_handler = logging.FileHandler(os.path.join(log_dir, f"{file_name}_{curr_partition}.log"), "w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S"))
        LOGGER.addHandler(file_handler)

    model_path = "/scratch/c.c21051562/resources/downloaded_models/Llama-3.1-8B-Instruct"
    LOGGER.info("Loading model %s", model_path)
    torch.cuda.reset_max_memory_allocated()
    generator = load_model(model_path)
    
    # interpret-cxr train dev test-public 
    data_path = "/scratch/c.c21051562/workspace/arrg_sentgen/outputs/interpret_cxr/raw.json"
    LOGGER.info("Loading data %s", data_path)
    docs = load_data(data_path)
    docs = partition_list(docs, curr_partition, max_partition)
    LOGGER.info("Loaded %s docs, partition %s of %s", len(docs), curr_partition, max_partition)
    LOGGER.info("Dataset start with %s", docs[0]["doc_key"])
    
    prompt_path = "/scratch/c.c21051562/workspace/arrg_sentgen/llm_prompt"
    prompt = load_prompt(prompt_path)
    LOGGER.info("Prompt:\n %s", prompt)

    output_dir = "/scratch/c.c21051562/workspace/arrg_sentgen/outputs/interpret_cxr"
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, f"llm_sent_splits_{curr_partition}_of_{max_partition}.json")
    # assert not os.path.exists(output_file_path)
    # if os.path.exists(output_file_path):
    #     os.remove(output_file_path)
    
    batch_info = []
    batch_size = 64
    LOGGER.info("Batch size: %s", batch_size)
    
    # 如果ARCCA一次处理不完时，需要重复运行脚本，此时则跳过已经处理的文件
    processed_ids = []
    if os.path.exists(output_file_path):
        with open(output_file_path, "r") as f:
            for line in f.readlines():
                processed_doc_sent = json.loads(line.strip())
                processed_ids.append(f"{processed_doc_sent['doc_key']}#{processed_doc_sent['sent_idx']}")
        
    with open(output_file_path, "a") as f:
        def process_batch(batch_info):
            batch_sents = [sent_str for (_, _, sent_str) in batch_info]
            batch_sent_splits = llm_process(generator, prompt, batch_sents)
            for (doc_key, sent_idx, sent_str), sent_splits in zip(batch_info, batch_sent_splits):
                write_jsonline_to(f, doc_key=doc_key, sent_idx=sent_idx, sent_splits=sent_splits, original_sent=sent_str)
        
        start = time.time()
        for data_idx, (doc_key, sent_idx, sent_str) in enumerate(doc_sent_generator(docs)):
            # Manually batching data: if a doc_sent exist, skip it; if a doc section is empty, write the empty data to file and go next
            if f"{doc_key}#{sent_idx}" in processed_ids:
                continue
            elif sent_idx is None :
                write_jsonline_to(f, doc_key=doc_key, sent_idx=None, sent_splits=[], original_sent="")
            else:
                batch_info.append((doc_key, sent_idx, sent_str))
            
            # When a batch is full, pass it to llm.
            if len(batch_info) == batch_size:
                process_batch(batch_info)
                batch_info = [] # Clear the list for the next batch
                
            if (data_idx + 1) % 5000 == 0:
                check_memory()
                end = time.time()
                LOGGER.info("Processed %s sentences (curr doc %s). Time elapsed: %s", data_idx + 1, doc_key, seconds_to_time_str(end - first_start))
        
        # Process the last batch
        if len(batch_info) > 0:
            process_batch(batch_info)
        
        end = time.time()
        LOGGER.info("Finished. Processed %s sentences. Time elapsed: %s.", data_idx + 1, seconds_to_time_str(end - first_start))
    
    # Final check.
    LOGGER.info("File check. %s lines in %s", count_file_lines(output_file_path), output_file_path)

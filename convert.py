import glob, json, os, argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--segment_len', type=int, default=254,
                        help='the length of each example')
    # we set this to be 254 instead of 256 because we want the input to be like: <control_code> input_ids <eos>
    parser.add_argument('--stride', type=int, default=10,
                        help='stride to split training examples')
    parser.add_argument('--dev_size', type=float, default=0.1,
                        help='split ratio of development set for each language')
    args = parser.parse_args()

    gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2", do_lower_case=False)
    segments = {}

    
    dataset = load_dataset("ArtifactAI/arxiv_python_research_code", split='train[:50%]')


    def process_batch(batch):
    # Initialize the tokenizer inside the worker function to ensure each process has its own instance
        results = []
        for example in batch:
            code_content = example
            encoded = gpt2_tok.encode(code_content, max_length=1024, truncation=True)
            for i in range(0, len(encoded), args.stride):
                seg = encoded[i:i + args.segment_len]
                results.append(json.dumps({"token_ids": seg, "label": "Python"}))
        return results


    num_processes = min(48, cpu_count())  

    # Define batch size based on your dataset size and memory considerations
    batch_size = len(dataset) // (10 * num_processes)  # Adjust based on dataset size and available memory

    #batch_size = 1000  # Adjust based on your system's memory capacity and CPU power
    batches = [dataset[i:i + batch_size]['code'] for i in range(0, len(dataset), batch_size)]

    # Process batches in parallel
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_batch, batches)

    segments = {"Python": [item for sublist in results for item in sublist]}

    
    train, dev = [], []
    for key in segments:
        # we don't shuffle before splitting because we want the train and dev to be very different (less overlapping)
        tr, de = train_test_split(segments[key], test_size=args.dev_size)
        train += tr
        dev += de
    
    
    
    to_path = "dataset/source_code/json"
    if not os.path.isdir(to_path):
        os.makedirs(to_path)

    with open(os.path.join(to_path, "train.jsonl"), "w") as f:
        f.write("\n".join(train))

    with open(os.path.join(to_path, "dev.jsonl"), "w") as f:
        f.write("\n".join(dev))

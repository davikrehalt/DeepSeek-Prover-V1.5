import os
import time
import torch.multiprocessing as mp
from llama_cpp import Llama

from prover.utils import AttrDict, MODEL_FORMAT

class GeneratorProcess(mp.Process):
    def __init__(self, local_rank, node_rank, model_path, task_queue, request_statuses, lock, args):
        super().__init__()
        self.local_rank = local_rank
        self.node_rank = node_rank
        self.model_path = model_path
        self.task_queue = task_queue
        self.request_statuses = request_statuses
        self.lock = lock
        self.args = args
        self.prompt_func = MODEL_FORMAT[args.mode]['prompt']
        self.output_func = MODEL_FORMAT[args.mode]['output']

    def run(self):
        seed = int(time.time()) % 1000 + (self.node_rank * 8 + self.local_rank) * 1000
        os.environ['LOCAL_RANK'] = str(self.local_rank)
        
        llm = Llama(
            model_path=self.model_path,
            n_ctx=4096,  # Adjust based on your needs
            n_threads=8,  # Adjust based on your Mac Studio's CPU
            seed=seed
        )

        while True:
            inputs = self.task_queue.get()
            if inputs is None:  # Terminate when receiving None
                break
            model_inputs = [
                ''.join([
                    item.get('_extra_header', str()),
                    self.prompt_func(item),
                    item.get('_extra_prompt', str()),
                ]) for _, _, item in inputs
            ]
            outputs = []
            for input_text in model_inputs:
                output = llm.create_completion(
                    input_text,
                    max_tokens=self.args.max_tokens,
                    temperature=self.args.temperature,
                    top_p=self.args.top_p,
                    stream=False,
                )
                outputs.append(self.output_func(output['choices'][0]['text']))

            with self.lock:
                for (_, request_id, _), output in zip(inputs, outputs):
                    self.request_statuses[request_id] = output

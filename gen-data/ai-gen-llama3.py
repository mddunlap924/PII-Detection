from utils import (load_cfg,
                   debugger_is_active)
from faker import Faker  # generates fake data
import ctypes
import argparse
import random
from pathlib import Path
from tqdm.auto import tqdm
import transformers
import numpy as np
import pandas as pd
import torch
import time
import gc
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

DEVICE = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device: {DEVICE}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"Pytorch {torch.__version__}")


# Seed the same seed to all
libc = ctypes.CDLL("libc.so.6")


def seed_everything(*, seed=42):
    Faker.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clear_memory():
    libc.malloc_trim(0)
    torch.cuda.empty_cache()
    gc.collect()


def load_model(model_path: str, *, quantize: bool = False):
    model_pipeline = transformers.pipeline(
        "text-generation",
        model=model_path,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="cuda",)
    return model_pipeline


def generate_texts(pipeline, generated_df, path_save):

    # Generate the texts
    for i in tqdm(range(len(generated_df))):
        start = time.time()
        # Get the prompt
        prompt = generated_df.prompt.iloc[i]
        max_new_tokens = generated_df['max_new_tokens'].iloc[i]
        temperature = generated_df['temperature'].iloc[i]
        top_p = generated_df['top_p'].iloc[i]
        top_k = int(generated_df['top_k'].iloc[i])
        repeat_penalty = generated_df['repetition_penalty'].iloc[i]
        file_name = generated_df['file_name'].iloc[i]
        writing_style = generated_df['writing_style'].iloc[i]
        fields_used = generated_df['fields_used'].iloc[i]

        # Tokenize the prompt
        prompt = pipeline.tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True
        )

        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        # Generate the outputs from prompt
        outputs = pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temperature,
        )
        # print(outputs[0]["generated_text"][len(prompt):])
        generated_df.loc[i, 'generated_text'] = outputs[0]["generated_text"]

        # Partial save of data
        if i % 5 == 0:
            generated_df.to_csv(path_save, index=False, encoding="UTF-8")
        print(
            f"Complete the text for {i}-th student {time.time() - start: .1f} seconds")
    # Save generated_df to csv
    generated_df.to_csv(path_save, index=False, encoding="UTF-8")
    print(f'Saved at: {path_save}')


label_types = ['NAME', 'EMAIL', 'USERNAME', 'ID_NUM',
               'PHONE_NUM', 'URL_PERSONAL', 'STREET_ADDRESS']

if __name__ == '__main__':

    # Determine if running in debug mode
    # If in debug manually point to CFG file
    is_debugger = debugger_is_active()

    # Construct the argument parser and parse the arguments
    if is_debugger:
        args = argparse.Namespace()
        args.dir = os.getenv('BASE_DIR') + '/gen-data/cfgs'
        args.name = 'cfg-auto-llama3-v0.yaml'
    else:
        arg_desc = '''This program points to input parameters for model training'''
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=arg_desc)
        parser.add_argument("-cfg_dir",
                            "--dir",
                            required=True,
                            help="Base Dir. for the YAML config. file")
        parser.add_argument("-cfg_filename",
                            "--name",
                            required=True,
                            help="File name of YAML config. file")
        args = parser.parse_args()
        print(args)
    # Load the configuration file
    CFG = load_cfg(base_dir=Path(args.dir),
                   filename=args.name)
    CFG.base_dir = os.getenv('BASE_DIR')
    CFG.gen_dir = os.getenv('GEN_DIR')
    CFG.llm_dir = os.getenv('LLM_MODELS')

    MODEL_PATH = str(Path(CFG.llm_dir) / CFG.model)
    print(f'MODEL_PATH: {MODEL_PATH}')

    # Seed everything
    seed_everything(seed=CFG.seed)

    # Path to save generated csv
    save_gen_filename = (f'gen_{CFG.prompt_folder}_{CFG.model}_'
                         f'N{CFG.generate_text.N}_{CFG.filename}.csv')

    # List of topics
    with open('./gen-data/prompt-templates/topics-list.txt') as f:
        topics = f.read()
    topics = topics.split('\n')

    # List of majors
    with open('./gen-data/prompt-templates/majors.txt') as f:
        majors = f.read()
    majors = majors.split('\n')

    # Generate Placeholder Text from LLM
    cols = ['IDENTIFICATION_NUM', 'STREET_ADDRESS', 'PHONE_NUM',
            'USERNAME', 'URL_PERSONAL', 'EMAIL']

    writing_style = [
        'an essay',
        'a critical analysis (with citations and references)',
        'an untitled blog (i.e., without a title) ',
        'a few paragraphs (without a title)'
    ]
    fields_used = []
    writing_styles = []
    for _ in range(CFG.generate_text.N):
        fields_to_use = random.sample(cols, random.randint(1, 2))
        random.shuffle(fields_to_use)
        fields_used.append(", ".join(['YOUR_NAME'] + fields_to_use))
        writing_styles.append(random.choice(writing_style))

    # Store in dataframe
    df = pd.DataFrame({'fields_used': fields_used,
                      'writing_style': writing_styles})
    del fields_to_use, fields_used, writing_styles

    # Generate model parameter settings
    df['max_new_tokens'] = [random.choice([2048]) for _ in range(len(df))]
    df['temperature'] = [random.choice(
        [10, 20, 30, 70]) / 100 for _ in range(len(df))]
    df['top_p'] = [random.randint(a=90, b=95) / 100 for _ in range(len(df))]
    df['top_k'] = [random.choice([40, 50]) for _ in range(len(df))]
    df['repetition_penalty'] = [random.choice(
        [1.1, 1.2]) for _ in range(len(df))]

    # Generate occupation
    df['occupation'] = [random.choice(majors).lower() for _ in range(len(df))]
    df['topic'] = [random.choice(topics).lower() for _ in range(len(df))]

    # Prompt fields to insert
    def prompt_placeholder(fields):
        fields = fields.split(', ')
        return '\n'.join(['{' + f'{field}' + '}' for field in fields])

    df['prompt_pii'] = df.apply(lambda x: prompt_placeholder(fields=x['fields_used']),
                                axis=1)

    # List of prompts
    prompt_files = {
        'mixed': (list(Path(f'./gen-data/prompt-templates/placeholder/mixed-llama3').glob('*.txt'))),
    }

    def create_prompt(files: dict, data: pd.Series):
        if random.random() >= 0.0:
            file = random.sample(files['mixed'], 1)[0]
        else:
            file = random.sample(files['names'], 1)[0]
        with open(file) as f:
            prompt = f.read()
        prompt = prompt.replace('{OCCUPATION}', data['occupation'])
        prompt = prompt.replace('{REPORT}', data['writing_style'])
        prompt = prompt.replace('{TOPIC}', data['topic'])

        system_prompt = prompt.split('%%%%%%%%%%%%%%%%%%%%%%%%%')[0].strip()
        user_prompt = prompt.split('%%%%%%%%%%%%%%%%%%%%%%%%%')[1].strip()

        prompt_defs = {
            'YOUR_NAME': "Full name",
            'IDENTIFICATION_NUM': "Online student identification number",
            'STREET_ADDRESS': "Home street address",
            'PHONE_NUM': "Personal phone number",
            'USERNAME': "Online student username",
            'URL_PERSONAL': "Personal website or social medial platform",
            'EMAIL': "Personal email address"}

        sys_pii = []
        for pii in data.prompt_pii.split('\n'):
            sys_pii.append(f'{pii}: {prompt_defs[pii[1:-1]]}')
        sys_pii = '\n'.join(sys_pii)

        system_prompt = system_prompt.replace('{INSERT_INFO_HERE}', sys_pii)
        user_prompt = user_prompt.replace(
            '{INSERT_INFO_HERE}', data['prompt_pii'])

        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return file.name, prompt

    # Create prompt for the model
    df['file_name'], df['prompt'] = zip(*df.apply(lambda x: create_prompt(files=prompt_files,
                                                                          data=x), axis=1))

    # Model used to gen. text
    df['model'] = CFG.model

    # Generate the text
    model = load_model(model_path=MODEL_PATH)
    generate_texts(pipeline=model,
                   generated_df=df,
                   path_save=str(Path(CFG.gen_dir) / 'placeholder' / save_gen_filename))

    print('End of Script - Complete')

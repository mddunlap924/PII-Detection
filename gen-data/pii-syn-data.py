from gendata import (tokenize_with_spacy,
                     assign_labels,
                     create_token_map,
                     verify_df)
from utils import (load_cfg,
                   debugger_is_active)
from load_data import LoadData
import create_datasets
import unicodedata
from typing import List
import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
from spacy.lang.en import English
from pathlib import Path
import random
import argparse
import sys
import ctypes
from faker import Faker  # generates fake data
import gc
import string
import re
import time
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'


# Custom (cx) modules

DEVICE = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device: {DEVICE}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"Pytorch {torch.__version__}")


# Ensure that all operations are deterministic on GPU (if used) for
# reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Seed the same seed to all
libc = ctypes.CDLL("libc.so.6")


def seed_everything(*, seed=42):
    Faker.seed(0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clear_memory():
    libc.malloc_trim(0)
    torch.cuda.empty_cache()
    gc.collect()


def generate_street_address(fake):
    if random.random() >= 0.0:
        sa = str(fake.address()).replace("\n", random.choice([", ", " "]))
    else:
        sa = str(fake.address())

    if random.random() >= 0.9:
        sa = {0: fake.street_name() + ', ' + fake.city(),
              1: fake.street_address(),
              2: fake.street_address(),
              3: fake.street_address(),
              4: fake.street_address()}
        sa = sa[random.choice([0, 1, 2, 3, 4])]
    return sa


# Random generate 12 random number
def get_userid(length=16):
    """Generate userid - """
    if random.random() >= 0.30:
        # very common in training data 034626995785
        userid = str(random.randint(10**11, 10**12 - 1))
    else:
        if random.random() >= 0.5:
            # DM:705244534902
            userid = ("".join(random.choices(string.ascii_uppercase, k=2)) +
                      ':' + str(random.randint(10**11, 10**12 - 1)))
        else:
            if random.random() >= 0.25:
                # nMFtUVxSUI|33529258
                userid = ("".join(random.choices(string.ascii_letters, k=10)) +
                          '|' + str(random.randint(10**8, 10**9 - 1)))
            else:
                # ras21 or 51,00,23,0
                userid = ("".join(random.choices(string.ascii_letters, k=random.randint(
                    3, 5))) + str(random.randint(10**4, 10**6 - 1)))
    # Split id_num
    if random.random() >= 1.0:
        sep = random.choice([' ', '-'])
        n = int(len(userid) / random.choice([2, 3]))
        userid = sep.join([userid[i:i + n] for i in range(0, len(userid), n)])

    return userid


# Unique combinations of first / last name
def combine_first_last(fn: str, ln: str, algo_num: int):

    initials = [i[0] for i in (fn + ' ' + ln).split(' ')]

    if algo_num == 0:
        fn = fn[0]
        ln = ln
    elif algo_num == 1:
        if len(fn.split(' ')) == 2:
            fn = fn.split(' ')[0][0] + fn.split(' ')[1][0]
        else:
            fn = fn[0] + random.choice(string.ascii_lowercase)
        ln = ln
    return fn, ln


def social_media(username, prob):
    social_media_platforms = {
        'LinkedIn': 'linkedin.com/',
        'YouTube': 'youtube.com/',
        'Instagram': 'instagram.com/',
        'GitHub': 'github.com/',
        'Facebook': 'facebook.com/',
        'Twitter': 'twitter.com/',
    }

    if random.random() >= prob:
        platform, domain = random.choice(
            list(social_media_platforms.items())[0:2])
    else:
        platform, domain = random.choice(
            list(social_media_platforms.items())[2:])

    if platform == 'YouTube':
        post = {
            0: f'channel/UC{"".join(random.choices(string.ascii_letters + string.digits, k=random.randint(12,14)))}',
            1: f'channel/watch?v={"".join(random.choices(string.ascii_letters + string.digits, k=random.randint(10,12)))}',
            2: f'channel/user/{username}',
            3: f'c/{username}',
        }
        fake_url = f'https://www.{domain}{post[random.randint(0, 3)]}'
    elif platform == 'LinkedIn':
        post = {0: f'in/{username}', 1: f'{username}'}
        if random.random() >= 0.50:
            fake_url = f'https://www.{domain}{post[0]}'
        else:
            fake_url = f'https://www.{domain}{post[1]}'

    else:
        fake_url = f'https://{domain}{username}'
    return fake_url


def personal_site(first_name, last_name, username):
    print()
    fake = Faker()
    uri_path = fake.uri_path()
    tld = fake.tld()
    uri_ext = fake.uri_extension()
    domain_word = {0: f'{first_name}-{last_name}',
                   1: f'{first_name}',
                   2: f'{last_name}',
                   3: f'{first_name}{last_name}',
                   4: f'{username}'}
    www = random.choice(['', 'www.'])
    fake_url = f'https://{www}{domain_word[random.randint(0, 4)]}.{tld}/{uri_path}{uri_ext}'
    return fake_url.replace(' ', '').lower()


# Generate the personal url from social media
def generate_fake_social_media_url(first_name, last_name, algo):

    if random.random() >= 0.50:
        first_name, last_name, _ = get_name()

    username = generate_username(first_name, last_name, algo, 0.95)

    if random.random() >= 0.5:
        fake_url = social_media(username, 0.30)
    else:
        fake_url = personal_site(first_name, last_name, username)
    return fake_url


def generate_username(first_name, last_name, algo, prob):
    """usernames are created from first_name and last_name"""

    if random.random() >= 0.50:
        first_name, last_name, _ = get_name()

    SEPS = [""]

    if algo is not None:
        first_name, last_name = combine_first_last(
            fn=first_name, ln=last_name, algo_num=algo)
    else:
        if len(first_name.split(' ')) > 1:
            first_name = first_name.split(' ')[0]

    if random.random() >= prob:
        username = f"{first_name.lower()}{last_name.lower()}{random.randint(1,999)}"
    else:
        username = f"{first_name}{last_name}"

    # Replace whitespaces with seps
    username = username.replace(' ', random.choice(SEPS)).lower()
    return username


def generate_email(first_name, last_name, faker, algo):
    """usernames are created from first_name and last_name"""
    if random.random() >= 0.50:
        first_name, last_name, _ = get_name()

    initials = ''.join(
        [i[0] for i in (first_name + ' ' + last_name).split(' ')]).lower()

    if len(initials) == 3:
        first_name = first_name.split(' ')[0]

    algo_num = random.choice([0, 1, None, None, None, None])
    fn, ln = first_name, last_name
    if algo_num == 0:
        fn = fn[0]
        ln = ln
    elif algo_num == 1:
        if len(fn.split(' ')) == 2:
            fn = fn.split(' ')[0][0] + fn.split(' ')[1][0]
        else:
            fn = fn[0] + random.choice(string.ascii_lowercase)
        ln = ln
    else:
        if len(initials) == 3:
            if random.random() >= 0.3:
                fn = first_name.split(' ')[0]
                ln = last_name
            else:
                fn = initials
                ln = ''
        else:
            fn = first_name
            ln = last_name
    first_name = fn
    last_name = ln

    # Select real email domains
    if random.random() >= 0.05:
        if random.random() >= 0.50:
            # Select from top 10
            domain_name = random.choice(EMAIL_DOMAINS[0:6])
        else:
            # Select from botom 90
            domain_name = random.choice(EMAIL_DOMAINS[6:])
    else:
        domain_name = faker.domain_name()

    if algo_num is None:
        sa = {
            0: f"{first_name.lower()}{last_name.lower()}@{domain_name}",
            1: f"{first_name.lower()}{last_name.lower()}{random.randint(1, 99)}@{domain_name}",
            2: f"{first_name.lower()}.{last_name.lower()}@{domain_name}",
            3: f"{first_name.lower()}.{last_name.lower()}{random.randint(1, 99)}@{domain_name}",
            4: f"{first_name.lower()}_{last_name.lower()}@{domain_name}",
            5: f"{first_name.lower()}_{last_name.lower()}{random.randint(1, 99)}@{domain_name}",
            6: f"{last_name.lower()}{random.randint(1, 99)}@{domain_name}",
            7: f"{last_name.lower()}@{domain_name}"}
        email = sa[random.choice([0, 1, 2, 3, 4, 5, 6, 7])]
    else:
        sa = {
            0: f"{first_name.lower()}{last_name.lower()}@{domain_name}",
            1: f"{first_name.lower()}{last_name.lower()}{random.randint(1, 99)}@{domain_name}",
            2: f"{last_name.lower()}{random.randint(1, 99)}@{domain_name}",
            3: f"{last_name.lower()}@{domain_name}"}
        email = sa[random.choice([0, 1, 2, 3])]

    # Replace whitespaces with seps
    email = email.replace(' ', '')

    return email


def get_name():
    # Select the student country to generate the user info based on the country
    COUNTRIES = ["en_US", "en_US", "en_US", "en_US", "en_US",
                 "en_US", "en_US", "en_US", "en_US", "en_US",
                 "en_US", "en_US", "en_US", "en_US", "en_US",
                 "it_IT", "es_ES", "fr_FR"]
    faker = Faker(random.choice(COUNTRIES))
    if random.randint(0, 100) >= 80:
        idx_first = random.randint(0, len(FIRSTNAME_REAL) - 1)
        first_name = FIRSTNAME_REAL[idx_first]
        FIRSTNAME_REAL.pop(idx_first)

        idx_last = random.randint(0, len(LASTNAME_REAL) - 1)
        last_name = LASTNAME_REAL[idx_last]
        LASTNAME_REAL.pop(idx_last)
        # real = True
    else:
        if random.random() >= 0.25:
            first_name = faker.first_name()
            last_name = faker.last_name()
        else:
            first_name = faker.first_name() + ' ' + faker.last_name()
            last_name = faker.last_name()

    # Remove special characters
    first_name = first_name.replace('-', ' ')
    last_name = last_name.replace('-', ' ')

    # Normalize unicode characters
    first_name = unicodedata.normalize(
        'NFKD', first_name).encode(
        'ascii', 'ignore').decode('utf-8')
    last_name = unicodedata.normalize(
        'NFKD', last_name).encode(
        'ascii', 'ignore').decode('utf-8')

    return first_name, last_name, faker


def generate_student_info():
    """Generates all the user info (name, eamil addresses, phone number, etc) together """
    first_name, last_name, faker = get_name()

    real = random.choice([True])
    # Select algorithm for combining first and last names
    algos = []
    for _ in range(3):
        if random.random() >= 0.25:
            algos.append(random.choices([0, 1], k=1)[0])
        else:
            algos.append(None)

    user_name = generate_username(first_name, last_name, algos[0], 0.80)
    fake_url = generate_fake_social_media_url(first_name, last_name, algos[1])
    fake_email = generate_email(first_name, last_name, faker, algos[2])
    street_address = generate_street_address(fake=faker)
    student = {}
    student['ID_NUM'] = get_userid()  # User ID
    student['NAME'] = first_name + " " + last_name
    student['EMAIL'] = fake_email
    student['USERNAME'] = user_name
    student['PHONE_NUM'] = faker.phone_number().replace(" ", "")
    student['URL_PERSONAL'] = fake_url
    student['STREET_ADDRESS'] = street_address
    del faker
    clear_memory()
#     print(student)
    return student


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
        args.name = 'cfg1.yaml'
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

    # Training data
    df_train = pd.read_json(
        Path(
            CFG.gen_dir) /
        'pii-detection-removal-from-educational-data/train.json')
    df_train = df_train.explode(
        ['tokens', 'trailing_whitespace', 'labels']).reset_index(drop=True)

    # Load Real Names
    dfgn = pd.read_parquet(Path(CFG.gen_dir) /
                           'real-names/given_names_data.parquet')
    dfgn['is_ascii'] = dfgn.given_name.apply(lambda x: str(x).isascii())
    dfgn['len_gn'] = dfgn.given_name.apply(lambda x: len(str(x)))
    dfgn['num_names'] = dfgn.given_name.apply(lambda x: len(str(x).split(' ')))
    dfgn = dfgn[(dfgn['len_gn'] >= dfgn['len_gn'].mean()) & (
        dfgn['is_ascii']) & (dfgn['num_names'] <= 2)].reset_index(drop=True)

    dfsn = pd.read_parquet(Path(CFG.gen_dir) /
                           'real-names/surnames_data.parquet')
    dfsn['is_ascii'] = dfsn.surname.apply(lambda x: str(x).isascii())
    dfsn['len_gn'] = dfsn.surname.apply(lambda x: len(str(x)))
    dfsn['num_names'] = dfsn.surname.apply(
        lambda x: len(str(x).split(' ')) == 1)
    dfsn = dfsn[(dfsn['len_gn'] >= dfsn['len_gn'].mean()) & (
        dfsn['is_ascii']) & (dfsn['num_names'])].reset_index(drop=True)

    # Random combination of given names and surname
    FIRSTNAME_REAL, LASTNAME_REAL = zip(*random.sample(list(zip(dfgn['given_name'].tolist(),
                                                                dfsn['surname'].tolist())),
                                                       50_000))
    FIRSTNAME_REAL = [i.replace('-', ' ') for i in list(FIRSTNAME_REAL)]
    LASTNAME_REAL = [i.replace('-', ' ') for i in list(LASTNAME_REAL)]
    del dfgn, dfsn
    _ = gc.collect()
    print(f'# of Real First Names: {len(FIRSTNAME_REAL):,}')
    print(f'# of Real Last Names: {len(LASTNAME_REAL):,}')

    # Load top email domains
    with open('./gen-data/top-domains.txt', 'r') as file:
        # Read the entire file content
        EMAIL_DOMAINS = file.read()
    EMAIL_DOMAINS = EMAIL_DOMAINS.split('\n')

    # Create Syn. PII Data
    TOTAL = 4000  # Generate 10,000
    students = []
    for i in tqdm(range(TOTAL)):
        students.append(generate_student_info())

    # Store results in dataframe
    df = pd.DataFrame(students)

    # Reset index
    df = df.reset_index(drop=True)
    # Save to the csv file
    df.to_csv(
        Path(
            CFG.gen_dir) /
        f"pii_syn_data.csv",
        index=False,
        encoding='UTF-8')
    print(f'{Path(CFG.gen_dir) /f"pii_syn_data.csv"}')
    print('End of Script - Complete')

# Generated from: cledmodel-translation-part.ipynb
# Converted at: 2026-08-25T06:49:26.508Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # Part 1: Dataset Translation (English → Hindi, Malayalam, French)
# 
# This notebook translates all 32,431 sentences from the MAVEN event detection dataset into three target languages using Meta's **NLLB-200** (No Language Left Behind) translation model.
# 
# **Languages:** Hindi (`hin_Deva`), Malayalam (`mal_Mlym`), French (`fra_Latn`)  
# **Model:** `facebook/nllb-200-distilled-600M`  
# **Hardware:** NVIDIA Tesla T4 (Kaggle)


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

import kagglehub
# kagglehub.dataset_download('<owner>/<dataset-slug>')

# ## 1. Load and Explore MAVEN Dataset


import json

TRAIN_FILE = "/kaggle/input/datasets/kishorer2k4/maven-dataset/train.jsonl"

with open(TRAIN_FILE, "r") as f:
    sample = json.loads(next(f))

print(sample.keys())

for k, v in sample.items():
    print(f"\n{k}")
    print(type(v))

    if isinstance(v, list):
        print("Length:", len(v))

        if len(v) > 0:
            print("First item:")
            print(v[0])

    else:
        print(v)

import json
from tqdm.auto import tqdm

docs = 0
sentences = 0

with open(TRAIN_FILE) as f:
    for line in tqdm(f):
        item = json.loads(line)

        docs += 1
        sentences += len(item["content"])

print("Documents :", docs)
print("Sentences :", sentences)

import json
from tqdm.auto import tqdm

TRAIN_FILE = "/kaggle/input/datasets/kishorer2k4/maven-dataset/train.jsonl"

sentences = []

with open(TRAIN_FILE) as f:
    for line in tqdm(f):
        doc = json.loads(line)

        for sent in doc["content"]:
            sentences.append(sent["sentence"])

print("Total sentences:", len(sentences))

print("\nSample:")
print(sentences[0])

lengths = [len(s.split()) for s in sentences]

print("Average words:", sum(lengths)/len(lengths))
print("Max words:", max(lengths))
print("Min words:", min(lengths))

total_words = sum(lengths)

print(f"Total words: {total_words:,}")
print(f"Approx million words: {total_words/1_000_000:.2f}")

# ## 2. Setup: Install Dependencies & Load Translation Model


!pip install -q transformers sentencepiece accelerate sacremoses

import transformers
import torch

print(transformers.__version__)
print(torch.__version__)
print(torch.cuda.is_available())

!nvidia-smi

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL = "facebook/nllb-200-distilled-600M"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

print("Model loaded successfully")

import time
import torch

def translate_batch(texts, target_lang, batch_size=16):

    tokenizer.src_lang = "eng_Latn"

    outputs = []

    for i in range(0, len(texts), batch_size):

        batch = texts[i:i+batch_size]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(model.device)

        generated = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang),
            max_new_tokens=256
        )

        outputs.extend(
            tokenizer.batch_decode(
                generated,
                skip_special_tokens=True
            )
        )

    return outputs

# ## 3. Test Translation (Sample Batch)


sample = sentences[:10]

start = time.time()

translated = translate_batch(
    sample,
    target_lang="hin_Deva"
)

elapsed = time.time() - start

print(f"Time: {elapsed:.2f} sec")

for i in range(3):
    print("\nEN :", sample[i])
    print("HI :", translated[i])

import json
from tqdm.auto import tqdm

TRAIN_FILE = "/kaggle/input/datasets/kishorer2k4/maven-dataset/train.jsonl"

sentences = []

with open(TRAIN_FILE) as f:
    for line in tqdm(f):
        doc = json.loads(line)

        for sent in doc["content"]:
            sentences.append(sent["sentence"])

print("Loaded", len(sentences), "sentences")

sample = sentences[:10]

import time

start = time.time()

translated = translate_batch(
    sample,
    target_lang="hin_Deva"
)

elapsed = time.time() - start

print("Time:", elapsed)

for i in range(3):
    print("\nEN:", sample[i])
    print("HI:", translated[i])

import torch

print("CUDA:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0))
print("Model device:", next(model.parameters()).device)

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model = model.to(device)

print(next(model.parameters()).device)

sample = sentences[:128]

start = time.time()

translated = translate_batch(
    sample,
    target_lang="hin_Deva",
    batch_size=128
)

elapsed = time.time() - start

print("Sentences:", len(sample))
print("Time:", elapsed)
print("Sent/sec:", len(sample)/elapsed)

# ## 4. Full Dataset Translation Pipeline


import json
import os
from tqdm.auto import tqdm

def translate_dataset(
    sentences,
    target_lang,
    output_file,
    checkpoint_every=1000,
    batch_size=128
):

    translated = []

    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            translated = json.load(f)

        print("Resuming from", len(translated))

    start_idx = len(translated)

    for i in tqdm(range(start_idx, len(sentences), batch_size)):

        batch = sentences[i:i+batch_size]

        results = translate_batch(
            batch,
            target_lang=target_lang,
            batch_size=batch_size
        )

        translated.extend(results)

        if len(translated) % checkpoint_every < batch_size:

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(
                    translated,
                    f,
                    ensure_ascii=False
                )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            translated,
            f,
            ensure_ascii=False
        )

    return translated

# ### 4.1 Hindi Translation


hindi_sentences = translate_dataset(
    sentences,
    target_lang="hin_Deva",
    output_file="/kaggle/working/maven_hindi.json",
    batch_size=128
)

print(len(hindi_sentences))
print(hindi_sentences[:3])

# ### 4.2 Verify Hindi Translations


import json

TRAIN_FILE = "/kaggle/input/datasets/kishorer2k4/maven-dataset/train.jsonl"

with open(TRAIN_FILE) as f:
    doc = json.loads(next(f))

print("Title:", doc["title"])

for ev in doc["events"][:5]:
    mention = ev["mention"][0]

    sid = mention["sent_id"]

    print("\nTYPE:", ev["type"])
    print("TRIGGER:", mention["trigger_word"])

    print("ENGLISH SENTENCE:")
    print(doc["content"][sid]["sentence"])

    print("\nHINDI SENTENCE:")
    print(hindi_sentences[sid])

import json
from collections import Counter

TRAIN_FILE = "/kaggle/input/datasets/kishorer2k4/maven-dataset/train.jsonl"

trigger_counter = Counter()

with open(TRAIN_FILE) as f:
    for line in f:
        doc = json.loads(line)

        for event in doc["events"]:
            for mention in event["mention"]:
                trigger_counter[mention["trigger_word"].lower()] += 1

print("Unique triggers:", len(trigger_counter))
print("\nTop 50 triggers:\n")

for trigger, count in trigger_counter.most_common(50):
    print(trigger, count)

import json
from collections import Counter

TRAIN_FILE = "/kaggle/input/datasets/kishorer2k4/maven-dataset/train.jsonl"

event_counter = Counter()

with open(TRAIN_FILE) as f:
    for line in f:
        doc = json.loads(line)

        for event in doc["events"]:
            event_counter[event["type"]] += len(event["mention"])

print("Number of event types:", len(event_counter))
print()

for event, count in event_counter.most_common(30):
    print(f"{event:25} {count}")

from collections import Counter
import json

event_counter = Counter()

with open(TRAIN_FILE) as f:
    for line in f:
        doc = json.loads(line)

        for event in doc["events"]:
            event_counter[event["type"]] += len(event["mention"])

counts = list(event_counter.values())

print("Classes:", len(counts))
print("Max:", max(counts))
print("Min:", min(counts))
print("Median:", sorted(counts)[len(counts)//2])

print("\nBottom 20 classes:\n")

for cls, cnt in sorted(event_counter.items(), key=lambda x: x[1])[:20]:
    print(cls, cnt)

import random

indices = random.sample(range(len(sentences)), 5)

for idx in indices:
    print("="*80)
    print("EN:")
    print(sentences[idx])

    print("\nHI:")
    print(hindi_sentences[idx])

# ### 4.3 Malayalam Translation


malayalam_sentences = translate_dataset(
    sentences,
    target_lang="mal_Mlym",
    output_file="/kaggle/working/maven_malayalam.json",
    batch_size=128
)

# ### 4.4 French Translation


french_sentences = translate_dataset(
    sentences,
    target_lang="fra_Latn",
    output_file="/kaggle/working/maven_french.json",
    batch_size=128
)

# ### 4.5 Verify All Translations


import json

for file in [
    "/kaggle/working/maven_hindi.json",
    "/kaggle/working/maven_malayalam.json",
    "/kaggle/working/maven_french.json"
]:
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(file, len(data))

import random

indices = random.sample(range(len(sentences)), 5)

for idx in indices:
    print("="*80)
    print("EN:")
    print(sentences[idx])

    print("\nML:")
    print(malayalam_sentences[idx])
# Generated from: cled-training-2.ipynb
# Converted at: 2026-08-25T06:48:08.276Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # Part 2: Cross-Lingual Event Detection Model
# 
# This notebook trains an **XLM-RoBERTa** classifier for multilingual event detection across 168 event types. It covers data merging, model training, evaluation, and a live news detection demo.
# 
# **Model:** `xlm-roberta-base` + Projection Head  
# **Results:** Macro F1 = 0.777, Micro F1 = 0.665  
# **Hardware:** NVIDIA Tesla T4 (Kaggle)


import json
from tqdm.auto import tqdm

TRAIN_FILE = "/kaggle/input/datasets/kishorer2k4/maven-translated/train.jsonl"

HINDI_FILE = "/kaggle/input/datasets/kishorer2k4/maven-translated/maven_hindi.json"
MALAYALAM_FILE = "/kaggle/input/datasets/kishorer2k4/maven-translated/maven_malayalam.json"
FRENCH_FILE = "/kaggle/input/datasets/kishorer2k4/maven-translated/maven_french.json"

with open(HINDI_FILE, "r", encoding="utf-8") as f:
    hindi = json.load(f)

with open(MALAYALAM_FILE, "r", encoding="utf-8") as f:
    malayalam = json.load(f)

with open(FRENCH_FILE, "r", encoding="utf-8") as f:
    french = json.load(f)

print(len(hindi), len(malayalam), len(french))

# ## 1. Load Translated Data & Merge with MAVEN


import json
from tqdm.auto import tqdm

merged = []

idx = 0

with open(TRAIN_FILE, "r", encoding="utf-8") as f:

    for line in tqdm(f):

        doc = json.loads(line)

        for sent_id, sent in enumerate(doc["content"]):

            merged.append({
                "doc_id": doc["id"],
                "sent_id": sent_id,

                "english": sent["sentence"],
                "hindi": hindi[idx],
                "malayalam": malayalam[idx],
                "french": french[idx]
            })

            idx += 1

print("Merged:", len(merged))

OUTPUT = "/kaggle/working/maven_multilingual.json"

with open(
    OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        merged,
        f,
        ensure_ascii=False
    )

print("Saved:", OUTPUT)

enhanced = []

idx = 0

with open(TRAIN_FILE, "r", encoding="utf-8") as f:

    for line in tqdm(f):

        doc = json.loads(line)

        for sent_id, sent in enumerate(doc["content"]):

            enhanced.append({
                "doc_id": doc["id"],
                "title": doc["title"],
                "sent_id": sent_id,

                "english": sent["sentence"],
                "hindi": hindi[idx],
                "malayalam": malayalam[idx],
                "french": french[idx],

                "events": doc["events"]
            })

            idx += 1

print(len(enhanced))

with open(
    "/kaggle/working/maven_multilingual_full.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        enhanced,
        f,
        ensure_ascii=False
    )

# ## 2. Setup: Install Dependencies & Load Models


!pip install -q transformers sentence-transformers accelerate

import transformers
import sentence_transformers
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

print("Transformers:", transformers.__version__)
print("SentenceTransformers:", sentence_transformers.__version__)
print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("Using:", device)

# XLM-R
tokenizer = AutoTokenizer.from_pretrained(
    "xlm-roberta-base"
)

encoder = AutoModel.from_pretrained(
    "xlm-roberta-base"
).to(device)

# LaBSE
labse = SentenceTransformer(
    "sentence-transformers/LaBSE",
    device=device
)

print("Tokenizer loaded")
print("Encoder loaded")
print("LaBSE loaded")

import json

MULTI_FILE = "/kaggle/working/maven_multilingual.json"

with open(MULTI_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Samples:", len(data))
print(data[0].keys())

# ## 2.1 Load Multilingual Data


from torch.utils.data import Dataset

class AlignmentDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        item = self.data[idx]

        return {
            "en": item["english"],
            "hi": item["hindi"]
        }

dataset = AlignmentDataset(data)

print(len(dataset))

from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)

import torch.nn as nn

class ProjectionHead(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 256)
        self.act = nn.GELU()

    def forward(self, x):

        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        return x

projection = ProjectionHead().to(device)

print("Projection loaded")

import torch

def mean_pool(last_hidden_state, attention_mask):

    mask = attention_mask.unsqueeze(-1).float()

    summed = (last_hidden_state * mask).sum(1)

    counts = mask.sum(1)

    return summed / counts

import torch.nn.functional as F

def encode_texts(texts):

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)

    outputs = encoder(**inputs)

    pooled = mean_pool(
        outputs.last_hidden_state,
        inputs["attention_mask"]
    )

    projected = projection(pooled)

    projected = F.normalize(
        projected,
        p=2,
        dim=1
    )

    return projected

# ## 3. Dataset Format Inspection


import json

with open("/kaggle/working/maven_multilingual_full.json") as f:
    data = json.load(f)

sample = data[0]

print("Keys:", sample.keys())
print("\nEvents Example:\n")
print(sample["events"][0])

# ## 4. Event Type Taxonomy (168 Classes)


event_types = set()

for item in data:
    for event in item["events"]:
        event_types.add(event["type"])

print("Classes:", len(event_types))
print(sorted(list(event_types))[:20])


# ## 5. Sentence-Level Event Label Assignment


from collections import defaultdict

sentence_labels = []

for row in data:

    labels = set()

    current_sent = row["sent_id"]

    for event in row["events"]:

        for mention in event["mention"]:

            if mention["sent_id"] == current_sent:
                labels.add(event["type"])

    sentence_labels.append(list(labels))

print("Examples")

for i in range(5):
    print(sentence_labels[i])

from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()

Y = mlb.fit_transform(sentence_labels)

print(Y.shape)
print(len(mlb.classes_))

for i in range(len(data)):
    data[i]["label"] = Y[i]

from torch.utils.data import Dataset

class EventDataset(Dataset):

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):

        row = self.rows[idx]

        return {
            "text": row["english"],
            "labels": row["label"]
        }

from sklearn.model_selection import train_test_split

train_data, val_data = train_test_split(
    data,
    test_size=0.1,
    random_state=42
)

train_ds = EventDataset(train_data)
val_ds = EventDataset(val_data)

print(len(train_ds))
print(len(val_ds))

from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_ds,
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=16
)

# ## 7. XLM-RoBERTa Event Classifier


import torch
import torch.nn as nn

class XLMREventClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = encoder

        self.dropout = nn.Dropout(0.1)

        self.classifier = nn.Linear(
            768,
            168
        )

    def forward(
        self,
        input_ids,
        attention_mask
    ):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls = outputs.last_hidden_state[:,0]

        cls = self.dropout(cls)

        logits = self.classifier(cls)

        return logits

model = XLMREventClassifier().to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5
)

# ## 8. Training Loop (3 Epochs)


from tqdm.auto import tqdm
import torch

EPOCHS = 3

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    loop = tqdm(train_loader)

    for batch in loop:

        texts = batch["text"]

        labels = batch["labels"].float().to(device)

        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        enc = {
            k:v.to(device)
            for k,v in enc.items()
        }

        optimizer.zero_grad()

        logits = model(**enc)

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        loop.set_description(
            f"Epoch {epoch+1}"
        )

        loop.set_postfix(
            loss=loss.item()
        )

    print(
        f"\nEpoch {epoch+1} Avg Loss:",
        total_loss / len(train_loader)
    )

# ## 9. Evaluation — F1 Scores


from sklearn.metrics import f1_score
import numpy as np

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():

    for batch in tqdm(val_loader):

        texts = batch["text"]

        labels = np.array(batch["labels"])

        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        enc = {
            k:v.to(device)
            for k,v in enc.items()
        }

        logits = model(**enc)

        probs = torch.sigmoid(
            logits
        ).cpu().numpy()

        preds = (probs > 0.5).astype(int)

        all_preds.append(preds)
        all_labels.append(labels)

all_preds = np.vstack(all_preds)
all_labels = np.vstack(all_labels)

macro_f1 = f1_score(
    all_labels,
    all_preds,
    average="macro",
    zero_division=0
)

micro_f1 = f1_score(
    all_labels,
    all_preds,
    average="micro",
    zero_division=0
)

print("Macro F1:", macro_f1)
print("Micro F1:", micro_f1)

torch.save(
    model.state_dict(),
    "/kaggle/working/xlmr_event_baseline.pt"
)

from collections import Counter
import numpy as np

freq = Y.sum(axis=0)

weights = 1 / np.sqrt(freq + 1)

weights = weights / weights.mean()

class_weights = torch.tensor(
    weights,
    dtype=torch.float32
).to(device)

print(class_weights.shape)

# ## 10. Class-Weighted Loss & Multilingual Training Setup



criterion = torch.nn.BCEWithLogitsLoss(
    pos_weight=class_weights
)

import random
from torch.utils.data import Dataset

class MultiLingualDataset(Dataset):

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):

        row = self.rows[idx]

        text = random.choice([
            row["english"],
            row["hindi"],
            row["malayalam"],
            row["french"]
        ])

        return {
            "text": text,
            "labels": row["label"]
        }

train_ds = MultiLingualDataset(train_data)

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(
    "xlm-roberta-base"
)

encoder = AutoModel.from_pretrained(
    "xlm-roberta-base"
)

class XLMREventClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = encoder

        self.dropout = nn.Dropout(0.1)

        self.classifier = nn.Linear(
            768,
            168
        )

    def forward(
        self,
        input_ids,
        attention_mask
    ):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls = outputs.last_hidden_state[:,0]

        cls = self.dropout(cls)

        return self.classifier(cls)

model = XLMREventClassifier().to(device)

model.load_state_dict(
    torch.load(
        "/kaggle/working/xlmr_event_baseline.pt",
        map_location=device
    )
)

model.eval()

print("Loaded")

event_names = list(mlb.classes_)

print(event_names[:10])
print(len(event_names))

import torch

def predict_events(
    text,
    threshold=0.5
):

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    enc = {
        k:v.to(device)
        for k,v in enc.items()
    }

    with torch.no_grad():

        logits = model(**enc)

        probs = torch.sigmoid(
            logits
        ).cpu().numpy()[0]

    predictions = []

    for i,p in enumerate(probs):

        if p >= threshold:

            predictions.append(
                (
                    event_names[i],
                    float(p)
                )
            )

    predictions.sort(
        key=lambda x:x[1],
        reverse=True
    )

    return predictions

def predict_topk(text, k=5):

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    enc = {
        key:v.to(device)
        for key,v in enc.items()
    }

    with torch.no_grad():

        logits = model(**enc)

        probs = torch.sigmoid(logits)[0]

    topk = torch.topk(probs, k)

    results = []

    for score, idx in zip(
        topk.values,
        topk.indices
    ):
        results.append(
            (
                mlb.classes_[idx.item()],
                float(score)
            )
        )

    return results

# ## 11. Build Event Prediction Database


event_db = []

for row in data:

    text = row["english"]

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    enc = {
        k:v.to(device)
        for k,v in enc.items()
    }

    with torch.no_grad():

        logits = model(**enc)

        probs = torch.sigmoid(logits)[0]

    topk = torch.topk(probs, 5)

    predictions = []

    for score, idx in zip(
        topk.values,
        topk.indices
    ):

        predictions.append({
            "event": mlb.classes_[idx.item()],
            "score": float(score)
        })

    event_db.append({
        "english": row["english"],
        "hindi": row["hindi"],
        "malayalam": row["malayalam"],
        "french": row["french"],
        "predictions": predictions
    })

print("Done:", len(event_db))

import json

with open(
    "/kaggle/working/event_database.json",
    "w"
) as f:

    json.dump(
        event_db,
        f,
        ensure_ascii=False
    )

print("Saved")

# ## 12. Usage & Live Demos


# ### 12.1 Fetch Multilingual News (GNews API)


!pip install -q gnews newspaper4k lxml_html_clean

from gnews import GNews

def fetch_news(topic, lang="en", max_results=10):
    """Fetch news articles for a topic in a given language."""

    google_news = GNews(
        language=lang,
        max_results=max_results,
        period="7d"
    )

    articles = google_news.get_news(topic)

    results = []

    for article in articles:

        results.append({
            "title": article.get("title", ""),
            "description": article.get("description", ""),
            "published": article.get("published date", ""),
            "publisher": article.get("publisher", {}).get("title", ""),
            "url": article.get("url", "")
        })

    return results

print("fetch_news ready")

LANG_MAP = {
    "english":   "en",
    "hindi":     "hi",
    "malayalam": "ml",
    "french":    "fr"
}

def fetch_multilingual_news(topic, max_per_lang=5):
    """Fetch news for a topic in all 4 languages."""

    all_news = {}

    for lang_name, lang_code in LANG_MAP.items():

        print(f"Fetching {lang_name} news...")

        try:
            articles = fetch_news(
                topic,
                lang=lang_code,
                max_results=max_per_lang
            )
            all_news[lang_name] = articles
            print(f"  Found {len(articles)} articles")

        except Exception as e:
            print(f"  Error: {e}")
            all_news[lang_name] = []

    return all_news

print("fetch_multilingual_news ready")

def detect_events_in_news(news_dict, top_k=5):
    """Run event detection on fetched multilingual news."""

    results = []

    for lang_name, articles in news_dict.items():

        for article in articles:

            text = article["title"]

            if not text.strip():
                continue

            preds = predict_topk(text, k=top_k)

            results.append({
                "language": lang_name,
                "title": text,
                "publisher": article["publisher"],
                "published": article["published"],
                "url": article["url"],
                "events": [
                    {"type": e, "score": round(s, 3)}
                    for e, s in preds
                ]
            })

    return results

print("detect_events_in_news ready")

def display_results(results):
    """Pretty-print event detection results."""

    current_lang = None

    for r in results:

        if r["language"] != current_lang:
            current_lang = r["language"]
            print("" + "#" * 80)
            print(f"  {current_lang.upper()}")
            print("#" * 80)

        print("" + "-" * 60)
        print(f"  {r["title"]}")
        print(f"  Source: {r["publisher"]} | {r["published"]}")
        print(f"  Events:")

        for event in r["events"]:
            bar = " " * int(event["score"] * 20)
            print(f"{event["type"]:30s} "f"{event["score"]:.3f} {bar}")

print("display_results ready")

# Install Hugging Face libraries for local LLM summarization
!pip install -q transformers accelerate


# ### 12.2 Local LLM Summarizer (Qwen)


import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

print("Loading local multilingual LLM (Qwen/Qwen2.5-1.5B-Instruct)...")
print("This model runs entirely on Kaggle GPU — no API keys, no internet needed.")

# Determine device and dtype automatically
if torch.cuda.is_available():
    _device_map = "auto"
    _torch_dtype = torch.float16
    print("Using CUDA GPU for inference.")
else:
    _device_map = None
    _torch_dtype = torch.float32
    print("Using CPU for inference (slower).")

try:
    _qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    _qwen_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype=_torch_dtype,
        device_map=_device_map
    )

    _summariser_pipe = pipeline(
        "text-generation",
        model=_qwen_model,
        tokenizer=_qwen_tokenizer
    )
    _local_llm_ready = True
    print("Qwen-2.5-1.5B-Instruct loaded successfully!")
except Exception as e:
    _local_llm_ready = False
    print(f"Error loading local model: {e}")
    print("Falling back to structured text summaries.")


def summarise_news(articles, event_type):
    """Summarise multilingual news locally using Qwen2.5-1.5B-Instruct."""

    # Group headlines by language
    by_lang = {}
    for a in articles:
        lang = a["language"]
        if lang not in by_lang:
            by_lang[lang] = []
        by_lang[lang].append(a)

    text_block = ""
    for lang, items in by_lang.items():
        text_block += f"\n--- {lang.upper()} ---\n"
        for a in items:
            text_block += f"- {a['title']} (Source: {a['publisher']})\n"

    # Structured fallback if LLM is unavailable
    def get_structured_fallback():
        lines = [f"Event: {event_type}"]
        lines.append(f"Matched {len(articles)} articles across {len(by_lang)} languages\n")
        for lang, items in by_lang.items():
            lines.append(f"  {lang.upper()}:")
            for a in items:
                lines.append(f"    \u2022 {a['title']} ({a['publisher']})")
        return "\n".join(lines)

    if not _local_llm_ready:
        return get_structured_fallback()

    prompt = (
        f"You are a multilingual news analyst.\n\n"
        f"The user searched for the event type: {event_type}\n\n"
        f"Below are news headlines in various languages: {', '.join(by_lang.keys())}.\n"
        f"All these articles were identified by an event detector as matching \\\"{event_type}\\\"\n\n"
        f"Headlines:\n{text_block}\n\n"
        f"Please provide a concise 3-4 sentence summary of what is happening globally regarding this event type, "
        f"grouping key themes across languages and noting any regional patterns. "
        f"Do not repeat the prompt. Keep it informative, objective, and neutral."
        f"Give the summary in french, malayalam and hindi languages"
        
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Be concise and write a high-quality summary."},
        {"role": "user", "content": prompt}
    ]

    try:
        outputs = _summariser_pipe(
            messages,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        summary = outputs[0]["generated_text"][-1]["content"].strip()
        return summary
    except Exception as e:
        print(f"Error during summarization: {e}")
        return get_structured_fallback()


print("Local summariser function ready")


# Map event types to Google News search keywords
EVENT_TO_QUERY = {
    "Catastrophe": "earthquake OR tsunami OR flood OR disaster",
    "Attack": "attack OR bombing OR assault",
    "Military_operation": "military operation OR airstrike OR war",
    "Protest": "protest OR demonstration OR rally",
    "Killing": "killing OR murder OR assassination",
    "Terrorism": "terrorism OR terrorist attack",
    "Destroying": "destruction OR destroyed OR demolition",
    "Death": "death OR died OR killed",
    "Hostile_encounter": "conflict OR clash OR battle",
    "Change_of_leadership": "election OR prime minister OR president elected",
    "Legal_rulings": "court ruling OR verdict OR sentenced",
    "Arrest": "arrested OR arrest OR detained",
    "Fire": "fire OR wildfire OR blaze",
}

def get_search_query(event_type):
    """Get a search query for an event type. Falls back to the event name itself."""

    return EVENT_TO_QUERY.get(
        event_type,
        event_type.replace("_", " ")
    )

print("Event-to-query mapping ready")
print(f"Pre-mapped events: {len(EVENT_TO_QUERY)}")
print("Unmapped events will use event name as search query")

def mode1_event_to_news(
    event_type,
    max_per_lang=5,
    min_score=0.3
):
    """
    Mode 1: Event Type → Multilingual News + Summary

    1. Convert event type to search query
    2. Fetch news in all 4 languages
    3. Run event detection on each headline
    4. Filter: keep only articles where the model detects the target event
    5. Summarise with local LLM
    """

    query = get_search_query(event_type)
    print(f"\nSearching for: \"{query}\"  (event: {event_type})")
    print("=" * 80)

    # Fetch
    news = fetch_multilingual_news(
        query,
        max_per_lang=max_per_lang
    )

    # Detect + Filter
    all_results = detect_events_in_news(news)

    filtered = []

    for r in all_results:

        for e in r["events"]:

            if (
                e["type"].lower() == event_type.lower()
                and e["score"] >= min_score
            ):
                r["matched_score"] = e["score"]
                filtered.append(r)
                break

    filtered.sort(
        key=lambda x: x["matched_score"],
        reverse=True
    )

    print(f"\nFetched {len(all_results)} articles → "
          f"{len(filtered)} match \"{event_type}\" (score ≥ {min_score})")

    if not filtered:
        print("No matching articles found.")
        return

    # Display
    print("\n" + "─" * 80)
    print("  MATCHING ARTICLES")
    print("─" * 80)

    for r in filtered:

        lang_tag = r["language"][:2].upper()

        print(
            f"  [{lang_tag}] {r["title"]}\n"
            f"       Source: {r["publisher"]} | {r["published"]}\n"
            f"       {event_type} score: {r["matched_score"]:.3f}"
        )

    # Summarise
    print("\n" + "─" * 80)
    print("  LLM SUMMARY")
    print("─" * 80 + "\n")

    summary = summarise_news(filtered, event_type)

    print(summary)

    return filtered

print("Mode 1 ready: Event → News + Summary")

def mode2_news_to_events(
    text,
    top_k=10,
    threshold=0.1
):
    """
    Mode 2: News Text (any language) → Event Classes

    Paste any headline or sentence in any language.
    The model predicts event types with confidence scores.
    """

    print(f"\nInput: {text}")
    print("=" * 80)

    # Threshold-based
    threshold_preds = predict_events(
        text,
        threshold=threshold
    )

    # Top-K
    topk_preds = predict_topk(
        text,
        k=top_k
    )

    print(f"\nDetected Events (score ≥ {threshold}):\n")

    if threshold_preds:

        for event, score in threshold_preds:

            bar = "█" * int(score * 30)
            marker = " ◄ HIGH" if score >= 0.5 else ""
            print(f"  {event:30s} {score:.3f} {bar}{marker}")

    else:
        print("  No events above threshold.\n")
        print(f"  Top-{top_k} predictions:\n")

        for event, score in topk_preds:

            bar = "█" * int(score * 30)
            print(f"  {event:30s} {score:.3f} {bar}")

    return threshold_preds or topk_preds

print("Mode 2 ready: News → Events")

# ### 12.3 Interactive CLI Demo


while True:

    print("\n" + "=" * 80)
    print("  CLED — Cross-Lingual Event Detection")
    print("=" * 80)
    print("\n  [1] Event Type → Fetch multilingual news + LLM summary")
    print("  [2] Paste news text → Detect event classes")
    print("  [3] Exit\n")

    choice = input("Select mode (1/2/3): ").strip()

    if choice == "1":

        print(f"\nAvailable event types (sample): ")
        print(", ".join(event_names[:20]))
        print(f"... ({len(event_names)} total)\n")

        event_type = input("Enter event type: ").strip()

        if not event_type:
            continue

        mode1_event_to_news(event_type)

    elif choice == "2":

        text = input("\nPaste news headline (any language): ").strip()

        if not text:
            continue

        mode2_news_to_events(text)

    elif choice == "3":
        print("\nGoodbye!")
        break

    else:
        print("Invalid choice. Enter 1, 2, or 3.")
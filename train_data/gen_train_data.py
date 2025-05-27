import csv
import random
import uuid
from pathlib import Path

output_path = Path("tokenized_posts_labeled.csv")
output_path.unlink(missing_ok=True)

def generate_training_sample(text, label):
    return {
        "PostID": str(uuid.uuid4()),
        "PostText": text,
        "tokens": text.split(),
        "label": label
    }

themes = {
    "tech": {
        "en": [
            "New breakthrough in {} announced by {}.",
            "{} reveals cutting-edge {} platform.",
            "{} launches upgraded version of its {}.",
            "{} reports major security flaw in {}.",
            "Experts discuss future of {} in {}."
        ],
        "ua": [
            "Новий прорив у сфері {} оголошено компанією {}.",
            "{} презентувала інноваційну платформу {}.",
            "{} запускає оновлену версію свого {}.",
            "{} повідомляє про серйозну уразливість у {}.",
            "Експерти обговорюють майбутнє {} в {}."
        ],
        "vars": {
            "fields": ["AI", "quantum computing", "blockchain", "cybersecurity", "robotics", "mobile tech"],
            "companies": ["Apple", "Samsung", "OpenAI", "Meta", "Tesla", "SoftServe"]
        }
    },
    "politics": {
        "en": [
            "{} meets with {} to discuss {}.",
            "New sanctions imposed due to {} conflict.",
            "Parliament passes controversial law on {}.",
            "{} warns about rising tension in {}.",
            "Government plans reforms in {} sector."
        ],
        "ua": [
            "{} зустрічається з {} для обговорення {}.",
            "Нові санкції введено через конфлікт у {}.",
            "Парламент ухвалив суперечливий закон щодо {}.",
            "{} попереджає про напруження в {}.",
            "Уряд планує реформи в галузі {}."
        ],
        "vars": {
            "topics": ["Ukraine", "NATO", "trade", "defense", "cybersecurity", "foreign affairs"],
            "leaders": ["President Zelenskyy", "Joe Biden", "Olaf Scholz", "Macron", "Sunak"]
        }
    },
    "science": {
        "en": [
            "Researchers from {} discover {} in {}.",
            "New study reveals link between {} and {}.",
            "Breakthrough in {} treatment published by {}.",
            "NASA's mission to {} reaches milestone.",
            "Data shows impact of {} on global {}."
        ],
        "ua": [
            "Дослідники з {} відкрили {} у сфері {}.",
            "Нове дослідження показує зв'язок між {} та {}.",
            "Прорив у лікуванні {} опублікований командою з {}.",
            "Місія NASA до {} досягла важливого етапу.",
            "Дані показують вплив {} на глобальне {}."
        ],
        "vars": {
            "fields": ["cancer", "genetics", "climate", "space", "astrophysics", "neuroscience"],
            "orgs": ["Kyiv University", "NASA", "Oxford", "MIT", "Harvard"]
        }
    },
    "entertainment": {
        "en": [
            "{} wins award at {} for best {}.",
            "Rumors about {} spark debate on social media.",
            "{} cancels tour amid {} concerns.",
            "Surprise appearance by {} at {} concert.",
            "{} announces new movie with {}."
        ],
        "ua": [
            "{} отримав нагороду на {} за найкращий {}.",
            "Чутки про {} викликають резонанс у соцмережах.",
            "{} скасовує тур через побоювання щодо {}.",
            "Несподівана поява {} на концерті {}.",
            "{} оголошує про новий фільм з {}."
        ],
        "vars": {
            "celebs": ["Taylor Swift", "Dua Lipa", "Okean Elzy", "Monatik", "Brad Pitt", "Lina Kostenko"],
            "topics": ["performance", "health", "production", "controversy", "tour"]
        }
    },
    "business": {
        "en": [
            "{} sees surge in stock value after {}.",
            "Market reacts to {} earnings report.",
            "New trade deal between {} and {} finalized.",
            "Experts predict recession in {} sector.",
            "Startups raise millions to develop {}."
        ],
        "ua": [
            "Акції {} різко зросли після {}.",
            "Ринок реагує на звіт про прибутки {}.",
            "Нову торгову угоду між {} та {} укладено.",
            "Експерти прогнозують рецесію в галузі {}.",
            "Стартапи залучають мільйони для розробки {}."
        ],
        "vars": {
            "companies": ["Google", "Amazon", "Rozetka", "Silpo", "Grammarly", "Ajax Systems"],
            "sectors": ["IT", "retail", "logistics", "fintech", "e-commerce"]
        }
    }
}

samples = []

for label, data in themes.items():
    for lang in ['en', 'ua']:
        templates = data[lang]
        for _ in range(50):
            template = random.choice(templates)
            vars = data["vars"]

            if label == "tech":
                text = template.format(random.choice(vars["fields"]), random.choice(vars["companies"]))
            elif label == "politics":
                text = template.format(random.choice(vars["leaders"]), random.choice(vars["leaders"]), random.choice(vars["topics"]))
            elif label == "science":
                text = template.format(random.choice(vars["orgs"]), random.choice(vars["fields"]), random.choice(vars["fields"]))
            elif label == "entertainment":
                text = template.format(random.choice(vars["celebs"]), random.choice(vars["celebs"]), random.choice(vars["topics"]))
            elif label == "business":
                text = template.format(random.choice(vars["companies"]), random.choice(vars["sectors"]))

            samples.append(generate_training_sample(text, label))

random.shuffle(samples)
with output_path.open("w", newline='', encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["PostID", "PostText", "tokens", "label"])
    writer.writeheader()
    for sample in samples:
        writer.writerow({
            "PostID": sample["PostID"],
            "PostText": sample["PostText"],
            "tokens": " ".join(sample["tokens"]),
            "label": sample["label"]
        })

print(f" Збережено {len(samples)} прикладів у {output_path}")

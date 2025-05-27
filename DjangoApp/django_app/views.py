from django.shortcuts import render, redirect
from .models import StreamStatus, ClassifiedMessage
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
import json
import os
from datetime import datetime
from collections import Counter

# Create your views here.

def stream_status(request):
    control = StreamStatus.objects.first()
    return render(request, "stream_status.html", {"control": control})

def toggle_stream(request):
    control = StreamStatus.objects.first()
    control.is_active = not control.is_active
    control.save()
    return redirect("stream_status")


def stream_metrics_json(request):
    metrics_path = os.path.join(settings.BASE_DIR, 'data', 'stream_metrics.json')

    with open(metrics_path, 'r', encoding='utf-8') as file:
        metrics = json.load(file)

    control = StreamStatus.objects.first()
    is_active = control.is_active if control else False

    if is_active:
        start_time_str = metrics.get("last_start_time") or metrics.get("start_time")
        if start_time_str:
            try:
                start_time = datetime.fromisoformat(start_time_str)
                uptime = int((datetime.utcnow() - start_time).total_seconds())
            except:
                uptime = 0
        else:
            uptime = 0
    else:
        uptime = metrics.get("uptime_seconds", 0)

    return JsonResponse({
        "received_count": metrics.get("received_count", 0),
        "classified_count": metrics.get("classified_count", 0),
        "uptime_seconds": uptime,
        "current_throughput": metrics.get("current_throughput", 0),
        "throughput_limit": metrics.get("throughput_limit", 0),
        "status": metrics.get("status", "unknown")
    })

@csrf_exempt
def update_config(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    try:
        config_path = os.path.join(settings.BASE_DIR, "data", "config.json")
        data = json.loads(request.body)
        config = {
            "max_message_length": int(data.get("max_message_length", 1000)),
            "throughput_limit": int(data.get("throughput_limit", 1))
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        return JsonResponse({"status": "saved"})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


def latest_messages_json(request):
    messages_path = os.path.join(settings.BASE_DIR, 'data', 'all_messages.json')
    categories_path = os.path.join(settings.BASE_DIR, 'data/models', 'label_mapping.json')
    meta_path = os.path.join(settings.BASE_DIR, 'data', 'category_metadata.json')

    messages = []
    categories = {}

    meta_map = {}
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta_map = json.load(f)

    try:
        with open(messages_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:
                    try:
                        message = json.loads(line)
                        messages.append(message)
                    except json.JSONDecodeError:
                        continue

    except FileNotFoundError:
        return JsonResponse({"error": "Файл не знайдено."}, status=404)
    except Exception as e:
        return JsonResponse({"error": f"Помилка зчитування файлу: {str(e)}"}, status=500)

    try:
        with open(categories_path, 'r', encoding='utf-8') as file:
            categories = json.load(file)
    except FileNotFoundError:
        return JsonResponse({"error": "Файл з категоріями не знайдено."}, status=404)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Формат категорій некоректний."}, status=500)

    latest_10 = messages[-10:]

    for msg in latest_10:
        category_id = str(msg.get("category"))  # ключі в categories — рядки
        msg["category_label"] = categories.get(category_id, "unknown")
        msg["category_color"] = meta_map.get(category_id, {}).get("color", "#cccccc")

    return JsonResponse({"messages":latest_10}, safe=False)



def category_settings_page(request):
    return render(request, "category_settings.html")


def get_category_data():
    labels_path = os.path.join(settings.BASE_DIR, 'data/models', 'label_mapping.json')
    meta_path = os.path.join(settings.BASE_DIR, 'data', 'category_metadata.json')

    with open(labels_path, 'r', encoding='utf-8') as f:
        labels = json.load(f)

    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        metadata = {}

    result = []
    for cat_id, label in labels.items():
        meta = metadata.get(cat_id, {})
        result.append({
            "id": cat_id,
            "label": label,
            "description": meta.get("description", ""),
            "color": meta.get("color", "#cccccc")
        })

    return result, metadata


def get_categories(request):
    try:
        categories, _ = get_category_data()
        return JsonResponse({"categories": categories})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def update_category_metadata(request):
    if request.method not in ['POST', 'PUT']:
        return JsonResponse({"error": "Only POST or PUT allowed"}, status=405)

    meta_path = os.path.join(settings.BASE_DIR, 'data', 'category_metadata.json')

    try:
        data = json.loads(request.body)
        cat_id = str(data["id"])
        description = data.get("description", "")
        color = data.get("color", "#cccccc")

        _, metadata = get_category_data()
        metadata[cat_id] = {
            "description": description,
            "color": color
        }

        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return JsonResponse({"message": "Metadata updated", "category": metadata[cat_id]})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def delete_category_metadata(request, cat_id):
    if request.method != 'DELETE':
        return JsonResponse({"error": "Only DELETE allowed"}, status=405)

    meta_path = os.path.join(settings.BASE_DIR, 'data', 'category_metadata.json')

    try:
        _, metadata = get_category_data()
        if cat_id in metadata:
            deleted = metadata.pop(cat_id)
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            return JsonResponse({"message": f"Metadata for {cat_id} deleted", "deleted": deleted})
        else:
            return JsonResponse({"error": "Metadata not found"}, status=404)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

def search_classified_messages(request):
    query = request.GET.get("q", "").lower()
    category = request.GET.get("category")

    file_path = os.path.join(settings.BASE_DIR, 'data', 'all_messages.json')
    labels_path = os.path.join(settings.BASE_DIR, 'data/models', 'label_mapping.json')

    try:
        with open(labels_path, "r", encoding="utf-8") as f:
            label_map = json.load(f)
    except:
        label_map = {}

    results = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                    text = msg.get("PostText", "").lower()
                    cat_id = str(msg.get("category"))
                    if query and query not in text:
                        continue
                    if category and category != cat_id:
                        continue
                    msg["category_label"] = label_map.get(cat_id, "unknown")
                    results.append(msg)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return JsonResponse({"messages": []})

    return JsonResponse({"messages": results[-100:]})

def message_search_page(request):
    return render(request, "message_search.html")


def performance_logs_json(request):
    log_path = os.path.join(settings.BASE_DIR, 'data', 'performance_logs.json')

    try:
        if os.path.exists(log_path):
            with open(log_path, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        else:
            logs = []
    except Exception as e:
        logs = [{
            "timestamp": "",
            "status": "error",
            "details": f"Error reading log file: {str(e)}"
        }]

    return JsonResponse({"logs": logs[-200:]})

def performance_logs_page(request):
    return render(request, "performance_logs.html")

def chart_throughput_data(request):
    path = os.path.join(settings.BASE_DIR, 'data', 'performance_logs.json')
    data = []

    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            try:
                logs = json.load(f)[-100:]
                for item in logs:
                    data.append({
                        "time": item.get("timestamp"),
                        "value": item.get("throughput", 0)
                    })
            except:
                pass

    return JsonResponse({"data": data})


def chart_category_distribution(request):

    msg_path = os.path.join(settings.BASE_DIR, 'data', 'all_messages.json')
    label_path = os.path.join(settings.BASE_DIR, 'data/models', 'label_mapping.json')
    meta_path = os.path.join(settings.BASE_DIR, 'data', 'category_metadata.json')

    label_map = {}
    if os.path.exists(label_path):
        with open(label_path, 'r', encoding='utf-8') as f:
            label_map = json.load(f)

    meta_map = {}
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta_map = json.load(f)

    counts = Counter()

    if os.path.exists(msg_path):
        with open(msg_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    msg = json.loads(line.strip())
                    cat_id = str(msg.get("category"))
                    counts[cat_id] += 1
                except:
                    continue

    result = []
    for cat_id, count in counts.items():
        label = label_map.get(cat_id, f"category_{cat_id}")
        color = meta_map.get(cat_id, {}).get("color", "#cccccc")
        result.append({
            "label": label,
            "value": count,
            "color": color
        })

    return JsonResponse({"data": result})


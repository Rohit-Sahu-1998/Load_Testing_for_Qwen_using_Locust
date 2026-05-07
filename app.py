# app.py

import os
import uuid
import time
import json
import re
import base64
import asyncio
import csv
import subprocess
from io import BytesIO
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo

from PIL import Image, ImageSequence
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pdf2image import convert_from_bytes
from openai import OpenAI


# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = "outputs_vllm"
os.makedirs(OUTPUT_DIR, exist_ok=True)

METRICS_LOG_FILE = os.path.join(OUTPUT_DIR, "metrics_log.jsonl")
GPU_METRICS_CSV_FILE = os.path.join(OUTPUT_DIR, "gpu_metrics_log.csv")

VLLM_BASE_URL = "http://127.0.0.1:8002/v1"
VLLM_API_KEY = "EMPTY"
VLLM_MODEL_NAME = "/home/rohit.sahu/Qwen_model/cpt_codes/quantized_model/Quantized_model_qwen_4bit"

CPU_THREAD_WORKERS = 4

PDF_DPI = 110
MAX_IMAGE_SIZE = 1152
MAX_TOKENS = 500

SUPPORTED_EXTENSIONS = (".pdf", ".tif", ".tiff")

PROMPT = """
You are a strict JSON extraction system.

Carefully read the document image and extract visible values.

Rules:
1. Return ONLY valid JSON.
2. Use double quotes for all JSON keys and string values.
3. Do not add markdown, explanation, comments, or extra text.
4. Do not use trailing commas.
5. If a field is visible, do NOT leave it blank.
6. If a field is truly missing, return "" or [].
7. Keep the exact JSON schema.
8. For numbers, do not use commas. Example: use 2694.38, not 2,694.38.

Return this exact JSON:
{
  "claimant_name": "",
  "claimant_number": "",
  "tax_id": "",
  "practice_address": "",
  "billing_address": "",
  "diagnosis_codes": [],
  "date_of_service": "",
  "cpt_codes": [],
  "charges": [],
  "units": [],
  "invoice_date": "",
  "invoice_number": "",
  "taxonomy": ""
}
""".strip()


# ============================================================
# APP STATE
# ============================================================

app = FastAPI(title="FastAPI + vLLM Internal Queue Backend")

jobs: Dict[str, Dict[str, Any]] = {}

cpu_executor = ThreadPoolExecutor(max_workers=CPU_THREAD_WORKERS)

vllm_client = OpenAI(
    api_key=VLLM_API_KEY,
    base_url=VLLM_BASE_URL,
)


# ============================================================
# RESPONSE MODELS
# ============================================================

class SubmitResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: Dict[str, Any] | None = None
    error: str | None = None
    metrics: Dict[str, Any] | None = None
    json_file_path: str | None = None


# ============================================================
# BASIC HELPERS
# ============================================================

def now() -> float:
    return time.time()


def current_timestamp() -> str:
    return datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(timespec="seconds")


def round_or_none(value):
    return None if value is None else round(value, 3)


def get_file_extension(filename: str) -> str:
    return os.path.splitext(filename.lower())[1]


def get_empty_schema() -> Dict[str, Any]:
    return {
        "claimant_name": "",
        "claimant_number": "",
        "tax_id": "",
        "practice_address": "",
        "billing_address": "",
        "diagnosis_codes": [],
        "date_of_service": "",
        "cpt_codes": [],
        "charges": [],
        "units": [],
        "invoice_date": "",
        "invoice_number": "",
        "taxonomy": ""
    }


def normalize_output(parsed: Dict[str, Any]) -> Dict[str, Any]:
    schema = get_empty_schema()

    if not isinstance(parsed, dict):
        return schema

    for key in schema:
        if key in parsed:
            schema[key] = parsed[key]

    for key in ["diagnosis_codes", "cpt_codes", "charges", "units"]:
        if not isinstance(schema[key], list):
            schema[key] = [] if schema[key] in ("", None) else [schema[key]]

    for key in [
        "claimant_name",
        "claimant_number",
        "tax_id",
        "practice_address",
        "billing_address",
        "date_of_service",
        "invoice_date",
        "invoice_number",
        "taxonomy",
    ]:
        if schema[key] is None:
            schema[key] = ""
        elif not isinstance(schema[key], str):
            schema[key] = str(schema[key])

    return schema


def is_empty_extraction(data: Dict[str, Any]) -> bool:
    for value in data.values():
        if isinstance(value, list) and len(value) > 0:
            return False
        if isinstance(value, str) and value.strip():
            return False
    return True


# ============================================================
# GPU METRICS HELPERS
# ============================================================

def get_gpu_stats() -> Dict[str, Any]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()

        first_gpu = output.splitlines()[0]
        gpu_util, mem_used, mem_total, temp, power = first_gpu.split(",")

        return {
            "gpu_util_percent": int(gpu_util.strip()),
            "vram_used_mb": int(mem_used.strip()),
            "vram_total_mb": int(mem_total.strip()),
            "gpu_temp_c": int(temp.strip()),
            "gpu_power_w": float(power.strip()),
        }

    except Exception as e:
        return {
            "gpu_util_percent": None,
            "vram_used_mb": None,
            "vram_total_mb": None,
            "gpu_temp_c": None,
            "gpu_power_w": None,
            "gpu_error": str(e),
        }


def init_gpu_csv_log():
    if not os.path.exists(GPU_METRICS_CSV_FILE):
        with open(GPU_METRICS_CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp",
                    "job_id",
                    "filename",
                    "file_type",
                    "status",
                    "stage",
                    "gpu_util_percent",
                    "vram_used_mb",
                    "vram_total_mb",
                    "gpu_temp_c",
                    "gpu_power_w",
                ],
            )
            writer.writeheader()


def log_gpu_metrics_csv(
    job_id: str,
    filename: str,
    file_type: str,
    status: str,
    stage: str,
    gpu_stats: Dict[str, Any],
):
    init_gpu_csv_log()

    row = {
        "timestamp": current_timestamp(),
        "job_id": job_id,
        "filename": filename,
        "file_type": file_type,
        "status": status,
        "stage": stage,
        "gpu_util_percent": gpu_stats.get("gpu_util_percent"),
        "vram_used_mb": gpu_stats.get("vram_used_mb"),
        "vram_total_mb": gpu_stats.get("vram_total_mb"),
        "gpu_temp_c": gpu_stats.get("gpu_temp_c"),
        "gpu_power_w": gpu_stats.get("gpu_power_w"),
    }

    with open(GPU_METRICS_CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writerow(row)


# ============================================================
# JSON REPAIR / PARSER
# ============================================================

def strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def repair_json_text(text: str) -> str:
    text = strip_code_fences(text or "")

    first = text.find("{")
    last = text.rfind("}")

    if first == -1:
        raise ValueError("No JSON object found")

    if last == -1 or last <= first:
        text = text[first:] + "}"
    else:
        text = text[first:last + 1]

    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    text = re.sub(
        r'(?<=[{,\s])([A-Za-z_][A-Za-z0-9_]*)\s*:',
        r'"\1":',
        text
    )

    text = text.replace("\n", " ").replace("\t", " ")

    return text.strip()


def safe_extract_json(text: str) -> tuple[bool, Dict[str, Any] | None, str | None]:
    try:
        cleaned = repair_json_text(text)
        parsed = json.loads(cleaned)
        return True, parsed, None
    except Exception as e:
        return False, None, str(e)


# ============================================================
# DOCUMENT CONVERSION
# ============================================================

def resize_image(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
    return image


def convert_pdf_to_images(file_bytes: bytes) -> List[Image.Image]:
    images = convert_from_bytes(file_bytes, dpi=PDF_DPI)
    return [resize_image(img) for img in images]


def convert_tiff_to_images(file_bytes: bytes) -> List[Image.Image]:
    images = []

    with Image.open(BytesIO(file_bytes)) as img:
        for page in ImageSequence.Iterator(img):
            page = page.convert("RGB")
            page.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
            images.append(page.copy())

    if not images:
        raise ValueError("No image pages found in TIFF file.")

    return images


def convert_document_to_images(file_bytes: bytes, filename: str) -> List[Image.Image]:
    ext = get_file_extension(filename)

    if ext == ".pdf":
        return convert_pdf_to_images(file_bytes)

    if ext in (".tif", ".tiff"):
        return convert_tiff_to_images(file_bytes)

    raise ValueError(f"Unsupported file type: {ext}")


async def convert_document_to_images_async(file_bytes: bytes, filename: str) -> List[Image.Image]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        cpu_executor,
        convert_document_to_images,
        file_bytes,
        filename
    )


def pil_image_to_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# ============================================================
# vLLM INFERENCE
# ============================================================

def extract_usage(response) -> Dict[str, Any]:
    usage = getattr(response, "usage", None)

    if usage is None:
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }

    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def call_vllm_qwen(image: Image.Image, prompt: str):
    image_url = pil_image_to_data_url(image)

    response = vllm_client.chat.completions.create(
        model=VLLM_MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
        max_tokens=MAX_TOKENS,
        temperature=0,
    )

    content = response.choices[0].message.content
    usage = extract_usage(response)

    return content, usage


async def call_vllm_qwen_async(image: Image.Image, prompt: str):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        call_vllm_qwen,
        image,
        prompt
    )


def summarize_token_usage(token_usage: Dict[str, Any]) -> Dict[str, Any]:
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    has_any_value = False

    for _, usage in token_usage.items():
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        page_total_tokens = usage.get("total_tokens")

        if prompt_tokens is not None:
            total_prompt_tokens += prompt_tokens
            has_any_value = True

        if completion_tokens is not None:
            total_completion_tokens += completion_tokens
            has_any_value = True

        if page_total_tokens is not None:
            total_tokens += page_total_tokens
            has_any_value = True

    if not has_any_value:
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }

    return {
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
    }


async def run_vllm_inference_on_images(images: List[Image.Image]) -> Dict[str, Any]:
    extracted_data = {}
    raw_outputs = {}
    token_usage = {}
    warnings = []
    had_warning = False

    for idx, image in enumerate(images, start=1):
        page_key = f"page_{idx}"

        raw_text, usage = await call_vllm_qwen_async(image, PROMPT)

        raw_outputs[page_key] = raw_text
        token_usage[page_key] = usage

        print("\n===== RAW vLLM OUTPUT =====")
        print(raw_text)
        print("===========================\n")

        print("\n===== TOKEN USAGE =====")
        print(json.dumps(usage, indent=2))
        print("=======================\n")

        ok, parsed, parse_error = safe_extract_json(raw_text)

        if not ok:
            print(f"[WARN] First JSON parse failed for {page_key}: {parse_error}")
            print("[INFO] Retrying vLLM once...")

            retry_raw_text, retry_usage = await call_vllm_qwen_async(image, PROMPT)

            raw_outputs[f"{page_key}_retry"] = retry_raw_text
            token_usage[f"{page_key}_retry"] = retry_usage

            print("\n===== RAW vLLM RETRY OUTPUT =====")
            print(retry_raw_text)
            print("=================================\n")

            print("\n===== RETRY TOKEN USAGE =====")
            print(json.dumps(retry_usage, indent=2))
            print("=============================\n")

            ok, parsed, retry_error = safe_extract_json(retry_raw_text)

            if not ok:
                had_warning = True
                warnings.append({
                    "page": page_key,
                    "type": "json_parse_failed",
                    "first_error": parse_error,
                    "retry_error": retry_error,
                })
                normalized = get_empty_schema()
            else:
                normalized = normalize_output(parsed)
        else:
            normalized = normalize_output(parsed)

        if is_empty_extraction(normalized):
            had_warning = True
            warnings.append({
                "page": page_key,
                "type": "empty_extraction",
                "message": "Model returned valid JSON but all fields are empty."
            })

        extracted_data[page_key] = normalized

    token_summary = summarize_token_usage(token_usage)

    return {
        "pages_processed": len(images),
        "extracted_data": extracted_data,
        "raw_outputs": raw_outputs,
        "token_usage": token_usage,
        "token_summary": token_summary,
        "warnings": warnings,
        "had_warning": had_warning,
    }


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_json_output(job_id: str, filename: str, result: Dict[str, Any]) -> str:
    base_name = os.path.splitext(os.path.basename(filename))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{base_name}_{job_id}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return out_path


def save_failed_raw(job_id: str, filename: str, error_text: str) -> str:
    base_name = os.path.splitext(os.path.basename(filename))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{base_name}_{job_id}_failed.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(error_text)

    return out_path


def finalize_metrics(
    job: Dict[str, Any],
    token_summary: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    metrics = {
        "queue_delay_sec": None,
        "document_conversion_time_sec": round_or_none(
            job["document_conversion_finished_at"] - job["document_conversion_started_at"]
            if job.get("document_conversion_finished_at") and job.get("document_conversion_started_at") else None
        ),
        "vllm_inference_time_sec": round_or_none(
            job["inference_finished_at"] - job["inference_started_at"]
            if job.get("inference_finished_at") and job.get("inference_started_at") else None
        ),
        "end_to_end_time_sec": round_or_none(
            job["completed_at"] - job["submitted_at"]
            if job.get("completed_at") and job.get("submitted_at") else None
        ),
        "tokens_per_sec": None,
    }

    if token_summary:
        completion_tokens = token_summary.get("completion_tokens")
        inference_time = metrics.get("vllm_inference_time_sec")

        if completion_tokens is not None and inference_time and inference_time > 0:
            metrics["tokens_per_sec"] = round(completion_tokens / inference_time, 3)

    return metrics


def log_metrics(
    job_id: str,
    filename: str,
    file_type: str,
    status: str,
    metrics: Dict[str, Any],
    token_summary: Dict[str, Any] | None = None,
    result: Dict[str, Any] | None = None,
    error: str | None = None,
):
    token_summary = token_summary or {}

    entry = {
        "timestamp": current_timestamp(),
        "job_id": job_id,
        "filename": filename,
        "file_type": file_type,
        "status": status,

        "queue_delay_sec": metrics.get("queue_delay_sec") if metrics else None,
        "document_conversion_time_sec": metrics.get("document_conversion_time_sec") if metrics else None,
        "vllm_inference_time_sec": metrics.get("vllm_inference_time_sec") if metrics else None,
        "end_to_end_time_sec": metrics.get("end_to_end_time_sec") if metrics else None,

        "prompt_tokens": token_summary.get("prompt_tokens"),
        "completion_tokens": token_summary.get("completion_tokens"),
        "total_tokens": token_summary.get("total_tokens"),

        "tokens_per_sec": metrics.get("tokens_per_sec") if metrics else None,

        "gpu_before_inference": metrics.get("gpu_before_inference") if metrics else None,
        "gpu_after_inference": metrics.get("gpu_after_inference") if metrics else None,

        "review_required": result.get("review_required") if result else None,
        "warnings": result.get("warnings") if result else None,
        "error": error,
    }

    with open(METRICS_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ============================================================
# BACKGROUND JOB PROCESSING
# ============================================================

async def process_job(job_id: str):
    job = jobs[job_id]

    try:
        print(f"[JOB] Started job {job_id}")

        job["status"] = "processing"
        job["processing_started_at"] = now()

        job["document_conversion_started_at"] = now()
        images = await convert_document_to_images_async(
            job["file_bytes"],
            job["filename"]
        )
        job["document_conversion_finished_at"] = now()

        gpu_before = get_gpu_stats()

        log_gpu_metrics_csv(
            job_id=job_id,
            filename=job["filename"],
            file_type=job["file_type"],
            status="processing",
            stage="before_inference",
            gpu_stats=gpu_before,
        )

        job["inference_started_at"] = now()
        result = await run_vllm_inference_on_images(images)
        job["inference_finished_at"] = now()

        gpu_after = get_gpu_stats()

        log_gpu_metrics_csv(
            job_id=job_id,
            filename=job["filename"],
            file_type=job["file_type"],
            status="processing",
            stage="after_inference",
            gpu_stats=gpu_after,
        )

        clean_result = {
            "pages_processed": result["pages_processed"],
            "extracted_data": result["extracted_data"],
            "warnings": result.get("warnings", []),
            "review_required": result.get("had_warning", False),
        }

        token_summary = result.get("token_summary", {})

        job["status"] = (
            "completed_with_warning"
            if result.get("had_warning")
            else "completed"
        )
        job["result"] = clean_result
        job["completed_at"] = now()
        job["metrics"] = finalize_metrics(job, token_summary=token_summary)

        job["metrics"]["gpu_before_inference"] = gpu_before
        job["metrics"]["gpu_after_inference"] = gpu_after

        job["json_file_path"] = save_json_output(
            job_id=job_id,
            filename=job["filename"],
            result=clean_result,
        )

        log_metrics(
            job_id=job_id,
            filename=job["filename"],
            file_type=job["file_type"],
            status=job["status"],
            metrics=job["metrics"],
            token_summary=token_summary,
            result=clean_result,
            error=None,
        )

        job["file_bytes"] = None

        print("\n====================================")
        print(f"[JOB] FINISHED: {job_id}")
        print(f"Status: {job['status']}")
        print(f"Saved JSON: {job['json_file_path']}")
        print(f"Metrics Log: {METRICS_LOG_FILE}")
        print(f"GPU CSV Log: {GPU_METRICS_CSV_FILE}")
        print("Token Summary:")
        print(json.dumps(token_summary, indent=2))
        print("GPU Before:")
        print(json.dumps(gpu_before, indent=2))
        print("GPU After:")
        print(json.dumps(gpu_after, indent=2))
        print("Metrics:")
        print(json.dumps(job["metrics"], indent=2))
        print("====================================\n")

    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        job["completed_at"] = now()
        job["metrics"] = finalize_metrics(job, token_summary=None)
        job["file_bytes"] = None

        gpu_error_stats = get_gpu_stats()
        job["metrics"]["gpu_error_stage"] = gpu_error_stats

        failed_path = save_failed_raw(job_id, job["filename"], str(e))
        job["json_file_path"] = failed_path

        log_gpu_metrics_csv(
            job_id=job_id,
            filename=job["filename"],
            file_type=job.get("file_type", "unknown"),
            status="failed",
            stage="error",
            gpu_stats=gpu_error_stats,
        )

        log_metrics(
            job_id=job_id,
            filename=job["filename"],
            file_type=job.get("file_type", "unknown"),
            status=job["status"],
            metrics=job["metrics"],
            token_summary=None,
            result=None,
            error=str(e),
        )

        print(f"\n[JOB] FAILED: {job_id}")
        print(f"Error saved to: {failed_path}")
        print(f"Metrics Log: {METRICS_LOG_FILE}")
        print(f"GPU CSV Log: {GPU_METRICS_CSV_FILE}")
        print(str(e)[:3000])


# ============================================================
# STARTUP / SHUTDOWN
# ============================================================

@app.on_event("startup")
async def startup_event():
    init_gpu_csv_log()
    print("[INFO] FastAPI started")
    print("[INFO] No FastAPI queue is used")
    print("[INFO] vLLM internal queue/scheduler will handle inference requests")
    print(f"[INFO] Using vLLM server: {VLLM_BASE_URL}")
    print(f"[INFO] vLLM model name: {VLLM_MODEL_NAME}")
    print(f"[INFO] Metrics log file: {METRICS_LOG_FILE}")
    print(f"[INFO] GPU metrics CSV file: {GPU_METRICS_CSV_FILE}")


@app.on_event("shutdown")
async def shutdown_event():
    cpu_executor.shutdown(wait=False)


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
async def root():
    return {
        "message": "FastAPI + vLLM internal queue backend is running",
        "queue_mode": "vllm_internal_queue",
        "vllm_url": VLLM_BASE_URL,
        "vllm_model": VLLM_MODEL_NAME,
        "metrics_log_file": METRICS_LOG_FILE,
        "gpu_metrics_csv_file": GPU_METRICS_CSV_FILE,
        "supported_extensions": list(SUPPORTED_EXTENSIONS),
        "possible_statuses": [
            "queued",
            "processing",
            "completed",
            "completed_with_warning",
            "failed",
        ],
    }


@app.post("/submit", response_model=SubmitResponse)
async def submit_file(file: UploadFile = File(...)):
    filename = file.filename or ""
    ext = get_file_extension(filename)

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only PDF, TIF, and TIFF files are supported."
        )

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    job_id = str(uuid.uuid4())

    jobs[job_id] = {
        "status": "queued",
        "result": None,
        "error": None,
        "metrics": None,
        "json_file_path": None,
        "file_bytes": file_bytes,
        "filename": filename,
        "file_type": ext.replace(".", ""),
        "submitted_at": now(),
        "processing_started_at": None,
        "document_conversion_started_at": None,
        "document_conversion_finished_at": None,
        "inference_started_at": None,
        "inference_finished_at": None,
        "completed_at": None,
    }

    asyncio.create_task(process_job(job_id))

    return SubmitResponse(
        job_id=job_id,
        status="queued",
        message="File accepted. Inference will be scheduled by vLLM internal queue.",
    )


@app.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job ID not found.")

    job = jobs[job_id]

    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        result=job["result"],
        error=job["error"],
        metrics=job["metrics"],
        json_file_path=job["json_file_path"],
    )

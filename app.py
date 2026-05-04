# app.py

import os
import uuid
import time
import json
import re
import base64
import asyncio
from io import BytesIO
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo

from PIL import Image
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

VLLM_BASE_URL = "http://127.0.0.1:8002/v1"
VLLM_API_KEY = "EMPTY"
VLLM_MODEL_NAME = "/home/rohit.sahu/Qwen_model/cpt_codes/quantized_model/Quantized_model_qwen_4bit"
#/home/rohit.sahu/Qwen_model/cpt_codes/quantized_model/Quantized_model_qwen_4bit
MAX_QUEUE_SIZE = 100
NUM_WORKERS = 4
PDF_THREAD_WORKERS = 4

PDF_DPI = 110
MAX_IMAGE_SIZE = 1152
MAX_TOKENS = 500

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

app = FastAPI(title="FastAPI + vLLM PDF Extraction API")

job_queue: asyncio.Queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
jobs: Dict[str, Dict[str, Any]] = {}

pdf_executor = ThreadPoolExecutor(max_workers=PDF_THREAD_WORKERS)

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
# IMAGE / PDF HELPERS
# ============================================================

def pil_image_to_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def resize_image(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
    return image


def convert_pdf_to_images(file_bytes: bytes) -> List[Image.Image]:
    images = convert_from_bytes(file_bytes, dpi=PDF_DPI)
    return [resize_image(img) for img in images]


async def convert_pdf_to_images_async(file_bytes: bytes) -> List[Image.Image]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        pdf_executor,
        convert_pdf_to_images,
        file_bytes
    )


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


def run_vllm_inference_on_images(images: List[Image.Image]) -> Dict[str, Any]:
    extracted_data = {}
    raw_outputs = {}
    token_usage = {}
    warnings = []
    had_warning = False

    for idx, image in enumerate(images, start=1):
        page_key = f"page_{idx}"

        raw_text, usage = call_vllm_qwen(image, PROMPT)

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

            retry_raw_text, retry_usage = call_vllm_qwen(image, PROMPT)

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


async def run_vllm_inference_async(images: List[Image.Image]) -> Dict[str, Any]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        run_vllm_inference_on_images,
        images
    )


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


def log_metrics(
    job_id: str,
    filename: str,
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
        "status": status,

        "queue_delay_sec": metrics.get("queue_delay_sec") if metrics else None,
        "pdf_conversion_time_sec": metrics.get("pdf_conversion_time_sec") if metrics else None,
        "vllm_inference_time_sec": metrics.get("vllm_inference_time_sec") if metrics else None,
        "end_to_end_time_sec": metrics.get("end_to_end_time_sec") if metrics else None,

        "prompt_tokens": token_summary.get("prompt_tokens"),
        "completion_tokens": token_summary.get("completion_tokens"),
        "total_tokens": token_summary.get("total_tokens"),

        "tokens_per_sec": metrics.get("tokens_per_sec") if metrics else None,

        "review_required": result.get("review_required") if result else None,
        "warnings": result.get("warnings") if result else None,
        "error": error,
    }

    with open(METRICS_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def finalize_metrics(
    job: Dict[str, Any],
    token_summary: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    metrics = {
        "queue_delay_sec": round_or_none(
            job["processing_started_at"] - job["submitted_at"]
            if job.get("processing_started_at") and job.get("submitted_at") else None
        ),
        "pdf_conversion_time_sec": round_or_none(
            job["pdf_conversion_finished_at"] - job["pdf_conversion_started_at"]
            if job.get("pdf_conversion_finished_at") and job.get("pdf_conversion_started_at") else None
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


# ============================================================
# WORKER
# ============================================================

async def worker(worker_id: int):
    while True:
        job_id = await job_queue.get()
        job = jobs[job_id]

        try:
            print(f"[WORKER {worker_id}] Started job {job_id}")

            job["status"] = "processing"
            job["processing_started_at"] = now()

            job["pdf_conversion_started_at"] = now()
            images = await convert_pdf_to_images_async(job["file_bytes"])
            job["pdf_conversion_finished_at"] = now()

            job["inference_started_at"] = now()
            result = await run_vllm_inference_async(images)
            job["inference_finished_at"] = now()

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
            job["json_file_path"] = save_json_output(
                job_id=job_id,
                filename=job["filename"],
                result=clean_result,
            )

            log_metrics(
                job_id=job_id,
                filename=job["filename"],
                status=job["status"],
                metrics=job["metrics"],
                token_summary=token_summary,
                result=clean_result,
                error=None,
            )

            job["file_bytes"] = None

            print("\n====================================")
            print(f"[WORKER {worker_id}] JOB FINISHED: {job_id}")
            print(f"Status: {job['status']}")
            print(f"Saved JSON: {job['json_file_path']}")
            print(f"Metrics Log: {METRICS_LOG_FILE}")
            print("Token Summary:")
            print(json.dumps(token_summary, indent=2))
            print("Metrics:")
            print(json.dumps(job["metrics"], indent=2))
            print("====================================\n")

        except Exception as e:
            job["status"] = "failed"
            job["error"] = str(e)
            job["completed_at"] = now()
            job["metrics"] = finalize_metrics(job, token_summary=None)
            job["file_bytes"] = None

            failed_path = save_failed_raw(job_id, job["filename"], str(e))
            job["json_file_path"] = failed_path

            log_metrics(
                job_id=job_id,
                filename=job["filename"],
                status=job["status"],
                metrics=job["metrics"],
                token_summary=None,
                result=None,
                error=str(e),
            )

            print(f"\n[WORKER {worker_id}] JOB FAILED: {job_id}")
            print(f"Error saved to: {failed_path}")
            print(f"Metrics Log: {METRICS_LOG_FILE}")
            print(str(e)[:3000])

        finally:
            job_queue.task_done()


# ============================================================
# STARTUP / SHUTDOWN
# ============================================================

@app.on_event("startup")
async def startup_event():
    app.state.worker_tasks = [
        asyncio.create_task(worker(i + 1)) for i in range(NUM_WORKERS)
    ]
    print(f"[INFO] Started {NUM_WORKERS} FastAPI workers")
    print(f"[INFO] Using vLLM server: {VLLM_BASE_URL}")
    print(f"[INFO] vLLM model name: {VLLM_MODEL_NAME}")
    print(f"[INFO] Metrics log file: {METRICS_LOG_FILE}")


@app.on_event("shutdown")
async def shutdown_event():
    for task in app.state.worker_tasks:
        task.cancel()
    pdf_executor.shutdown(wait=False)


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
async def root():
    return {
        "message": "FastAPI + vLLM backend is running",
        "vllm_url": VLLM_BASE_URL,
        "vllm_model": VLLM_MODEL_NAME,
        "metrics_log_file": METRICS_LOG_FILE,
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
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    if job_queue.full():
        raise HTTPException(
            status_code=429,
            detail="Queue is full. Try again later."
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
        "filename": file.filename,
        "submitted_at": now(),
        "processing_started_at": None,
        "pdf_conversion_started_at": None,
        "pdf_conversion_finished_at": None,
        "inference_started_at": None,
        "inference_finished_at": None,
        "completed_at": None,
    }

    await job_queue.put(job_id)

    return SubmitResponse(
        job_id=job_id,
        status="queued",
        message="File accepted and added to queue",
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
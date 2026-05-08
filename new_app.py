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
MAX_TOKENS = 800

SUPPORTED_EXTENSIONS = (".pdf", ".tif", ".tiff")

# Options: cheque / comp_check / patient
DOCUMENT_TYPE = "cheque"


# ============================================================
# PROMPTS
# ============================================================

CHEQUE_PROMPT = """
You are extracting data from a cheque image.

Return ONLY a valid JSON object.
No explanation.
No markdown.
No backticks.

Required JSON structure:
{
  "check_number": null,
  "check_amount": null,
  "pay_to": null,
  "provider_name": null
}

Rules:
- check_number: read from MICR bottom line first, else from printed number.
- check_number must be digits only.
- Strip leading zeros.
- Maximum 9 digits. If longer than 9 digits, keep only the last 9.
- check_amount: dollar amount exactly as printed.
- pay_to: name written immediately after "Pay to the Order of".
- provider_name: company/provider who issued the cheque, not the bank.
- Any field not found must be null.
"""

COMP_CHECK_PROMPT = """
Look at every part of this image very carefully.
Include handwritten text, stamps, annotations, or printed text anywhere on the page.

Does this image contain the words "Comp Benefits" or "Comp Benefit" anywhere?

Examples that count:
- ATTN: Comp Benefits
- RE: Comp Benefits
- Comp Benefits Plan
- Workers Comp Benefits

Reply with ONLY one word:
YES or NO
"""

PATIENT_PROMPT = """
You are extracting patient/claim records from this document image.

Return ONLY valid JSON.
No markdown.
No explanation.
No backticks.

IMPORTANT:
- If a value is missing, use null.
- Never copy instruction text into field values.
- Never copy example placeholder text into field values.
- Extract only values that are actually visible in the document.

The page may contain records in any of these formats:

FORMAT A — Structured table:
Columns may include:
Tax ID, First Name, Last Name, Birthdate, Subscriber#, Group#,
Admit Date, Disch Date, Last ICN, Total Charges, Payments,
Refunding, Account Num, Reason.

FORMAT B — Single inline line per patient:
Example:
GRAVES, CHARLES  1/21/2026  FACILITY TERMED WITH MNS  397.18

Parse as:
patient name | date as dos | description text as reason_for_refund | trailing amount as claim_payment_amount.

FORMAT C — Paragraph or letter format:
Fields may be labelled:
Patient Name, Member ID, MID, DOS, Claim #, Reason, Amount, Your Share.

FORMAT D — Handwritten notes in any layout.

FORMAT E — Description column:
Example:
FARMAR SUSAN D - 1991228713 claim represented ICN -820253250532028

Parse as:
patient name | ICN/claim number if valid | amount from net amount if visible.

CRITICAL COMP BENEFITS RULE:
If the page contains "Comp Benefits" or "Comp Benefit" anywhere,
typed, printed, stamped, or handwritten, return exactly:
{
  "comp_benefits_detected": true,
  "has_patient_data": false,
  "patients": []
}

Do NOT extract patient data from such a page.

ANTI-HALLUCINATION RULES:
1. patient_name must be a real human patient name.
   Never use payer names, insurance names, bank names, or organization names.
   Invalid examples: Humana, Aetna, Medicare, Blue Cross, bank names.
   If no real human patient name is visible, use null.

2. claim_number must be exactly 15 digits.
   Corrected Claim can also be used if it is exactly 15 digits.
   Any other length must be null.
   Never use Group#, Tax ID, Provider ID, or Account Num as claim_number.

3. member_id can come from:
   Subscriber#, Subscriber ID, MID, Member ID, Patient ID,
   Humana ID number, Policy ID#, Policy Number, INS ID, ID#,
   Patient Acct. Number.
   Never use Group#.

4. claim_payment_amount can come from:
   Overpaid Amount, Amount, Correct Amnt, Net Amnt, Your Share,
   Refunding, Total Charges.
   If Correct Amnt is visible, prefer Correct Amnt.
   Include $ symbol if printed.

5. Extract every visible patient row or record on this page.
   Do not skip visible patient records.

6. For FORMAT B:
   name before date = patient_name
   date = dos
   trailing number = claim_payment_amount
   middle text = reason_for_refund

7. Patient name splitting:
   LAST, FIRST format:
     last_name = text before comma
     first_name = text after comma
   FIRST LAST format:
     split into first_name and last_name only if clear.

8. Every missing field must be null.
   Never omit a key.
   Never invent values.

9. If page has no patient data, return:
{
  "comp_benefits_detected": false,
  "has_patient_data": false,
  "patients": []
}

Return this JSON structure:
{
  "comp_benefits_detected": false,
  "has_patient_data": false,
  "patients": [
    {
      "patient_name": null,
      "first_name": null,
      "last_name": null,
      "dos": null,
      "member_id": null,
      "claim_number": null,
      "reason_for_refund": null,
      "claim_payment_amount": null,
      "payee_name": null
    }
  ]
}
"""

PROMPT_MAP = {
    "cheque": CHEQUE_PROMPT,
    "comp_check": COMP_CHECK_PROMPT,
    "patient": PATIENT_PROMPT,
}

PROMPT = PROMPT_MAP[DOCUMENT_TYPE]


# ============================================================
# APP STATE
# ============================================================

app = FastAPI(title="FastAPI + vLLM Document Extraction API")

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


# ============================================================
# PLACEHOLDER CLEANING
# ============================================================

PLACEHOLDER_PHRASES = [
    "full name exactly as written",
    "first name only",
    "last name only",
    "date of service",
    "admit date",
    "disch date",
    "value from subscriber",
    "member id",
    "patient id",
    "policy id",
    "exactly 15-digit",
    "never use group",
    "full text from reason",
    "reason for refund",
    "description text after patient name",
    "dollar amount from amount",
    "your share",
    "correct amnt",
    "refund",
    "total charges",
    "provider/payee name",
    "null if not found",
    "if clearly separable",
    "include $ if printed",
]


def clean_placeholder_value(value):
    if value is None:
        return None

    if not isinstance(value, str):
        value = str(value)

    value = value.strip()

    if value in ("", "null", "None", "NONE", "NULL"):
        return None

    lower_value = value.lower()

    for phrase in PLACEHOLDER_PHRASES:
        if phrase in lower_value:
            return None

    return value


# ============================================================
# SCHEMA HELPERS
# ============================================================

def get_empty_result_schema() -> Dict[str, Any]:
    if DOCUMENT_TYPE == "cheque":
        return {
            "check_number": None,
            "check_amount": None,
            "pay_to": None,
            "provider_name": None,
        }

    if DOCUMENT_TYPE == "comp_check":
        return {
            "comp_benefits_detected": False,
        }

    if DOCUMENT_TYPE == "patient":
        return {
            "comp_benefits_detected": False,
            "has_patient_data": False,
            "patients": [],
        }

    return {}


def normalize_cheque_output(parsed: Dict[str, Any]) -> Dict[str, Any]:
    schema = get_empty_result_schema()

    if not isinstance(parsed, dict):
        return schema

    for key in schema:
        schema[key] = clean_placeholder_value(parsed.get(key))

    if schema["check_number"]:
        digits = re.sub(r"\D", "", schema["check_number"])
        digits = digits.lstrip("0")
        if len(digits) > 9:
            digits = digits[-9:]
        schema["check_number"] = digits if digits else None

    return schema


def normalize_patient_output(parsed: Dict[str, Any]) -> Dict[str, Any]:
    schema = get_empty_result_schema()

    if not isinstance(parsed, dict):
        return schema

    comp_detected = parsed.get("comp_benefits_detected", False)
    has_patient_data = parsed.get("has_patient_data", False)
    patients = parsed.get("patients", [])

    schema["comp_benefits_detected"] = bool(comp_detected)
    schema["has_patient_data"] = bool(has_patient_data)

    if schema["comp_benefits_detected"]:
        schema["has_patient_data"] = False
        schema["patients"] = []
        return schema

    if not isinstance(patients, list):
        patients = []

    normalized_patients = []

    patient_schema_keys = [
        "patient_name",
        "first_name",
        "last_name",
        "dos",
        "member_id",
        "claim_number",
        "reason_for_refund",
        "claim_payment_amount",
        "payee_name",
    ]

    for patient in patients:
        if not isinstance(patient, dict):
            continue

        normalized = {}

        for key in patient_schema_keys:
            normalized[key] = clean_placeholder_value(patient.get(key))

        claim_number = normalized.get("claim_number")
        if claim_number:
            digits = re.sub(r"\D", "", claim_number)
            normalized["claim_number"] = digits if len(digits) == 15 else None

        # If all fields are null, skip this fake/empty patient row.
        if all(value is None for value in normalized.values()):
            continue

        normalized_patients.append(normalized)

    schema["patients"] = normalized_patients
    schema["has_patient_data"] = len(normalized_patients) > 0

    return schema


def normalize_comp_check_output(raw_text: str) -> Dict[str, Any]:
    text = (raw_text or "").strip().upper()
    return {
        "comp_benefits_detected": text.startswith("YES")
    }


def normalize_output(parsed: Dict[str, Any], raw_text: str = "") -> Dict[str, Any]:
    if DOCUMENT_TYPE == "cheque":
        return normalize_cheque_output(parsed)

    if DOCUMENT_TYPE == "patient":
        return normalize_patient_output(parsed)

    if DOCUMENT_TYPE == "comp_check":
        return normalize_comp_check_output(raw_text)

    return parsed if isinstance(parsed, dict) else {}


def is_empty_extraction(data: Dict[str, Any]) -> bool:
    if DOCUMENT_TYPE == "comp_check":
        return False

    if DOCUMENT_TYPE == "cheque":
        return all(value in ("", None, [], {}) for value in data.values())

    if DOCUMENT_TYPE == "patient":
        return (
            data.get("comp_benefits_detected") is False
            and data.get("has_patient_data") is False
            and data.get("patients") == []
        )

    return not bool(data)


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
                    "document_type",
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
        "document_type": DOCUMENT_TYPE,
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
        text,
    )

    text = text.replace("\n", " ").replace("\t", " ")

    return text.strip()


def safe_extract_json(text: str) -> tuple[bool, Dict[str, Any] | None, str | None]:
    if DOCUMENT_TYPE == "comp_check":
        return True, normalize_comp_check_output(text), None

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
        filename,
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
        prompt,
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
                normalized = get_empty_result_schema()
            else:
                normalized = normalize_output(parsed, retry_raw_text)
        else:
            normalized = normalize_output(parsed, raw_text)

        if is_empty_extraction(normalized):
            had_warning = True
            warnings.append({
                "page": page_key,
                "type": "empty_extraction",
                "message": "Model returned valid response but all fields are empty.",
            })

        extracted_data[page_key] = normalized

    token_summary = summarize_token_usage(token_usage)

    return {
        "pages_processed": len(images),
        "document_type": DOCUMENT_TYPE,
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
    out_path = os.path.join(OUTPUT_DIR, f"{base_name}_{DOCUMENT_TYPE}_{job_id}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return out_path


def save_failed_raw(job_id: str, filename: str, error_text: str) -> str:
    base_name = os.path.splitext(os.path.basename(filename))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{base_name}_{DOCUMENT_TYPE}_{job_id}_failed.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(error_text)

    return out_path


def finalize_metrics(
    job: Dict[str, Any],
    token_summary: Dict[str, Any] | None = None,
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
        "document_type": DOCUMENT_TYPE,
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
            job["filename"],
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
            "document_type": DOCUMENT_TYPE,
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
    print(f"[INFO] Document type: {DOCUMENT_TYPE}")
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
        "message": "FastAPI + vLLM document extraction backend is running",
        "queue_mode": "vllm_internal_queue",
        "document_type": DOCUMENT_TYPE,
        "vllm_url": VLLM_BASE_URL,
        "vllm_model": VLLM_MODEL_NAME,
        "metrics_log_file": METRICS_LOG_FILE,
        "gpu_metrics_csv_file": GPU_METRICS_CSV_FILE,
        "supported_extensions": list(SUPPORTED_EXTENSIONS),
        "possible_document_types": list(PROMPT_MAP.keys()),
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
            detail="Only PDF, TIF, and TIFF files are supported.",
        )

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
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
        "document_type": DOCUMENT_TYPE,
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
        message=f"File accepted. Document type: {DOCUMENT_TYPE}. Inference will be scheduled by vLLM internal queue.",
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

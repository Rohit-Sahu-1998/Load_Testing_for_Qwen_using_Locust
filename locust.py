import time
import random
from locust import HttpUser, task, between, events


PDF_PATHS = [
    #"/home/rohit.sahu/Qwen_model/samples_nonstandard_data/Document_1.pdf",
    #"/home/rohit.sahu/Qwen_model/samples_nonstandard_data/Document_5.pdf",
    #"/home/rohit.sahu/Qwen_model/samples_nonstandard_data/Document_4.pdf"
    # Add more PDFs here:
    #"/home/rohit.sahu/Qwen_model/samples_nonstandard_data/Document_2.pdf",
    #"/home/rohit.sahu/Qwen_model/samples_nonstandard_data/Document_3.pdf",
    #"/home/rohit.sahu/Qwen_model/temp_upload_Document 7.pdf",
    "/home/rohit.sahu/Qwen_model/cpt_codes/vllm_based_locust/output.pdf"
]

@events.request.add_listener
def request_listener(
    request_type,
    name,
    response_time,
    response_length,
    response,
    context,
    exception,
    start_time,
    url,
    **kwargs,
):
    if exception:
        print(f"[ERROR] {request_type} {name} -> {exception}")


class QwenApiUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def submit_and_poll(self):
        pdf_path = random.choice(PDF_PATHS)

        try:
            with open(pdf_path, "rb") as f:
                with self.client.post(
                    "/submit",
                    files={
                        "file": (
                            pdf_path.split("/")[-1],
                            f,
                            "application/pdf",
                        )
                    },
                    catch_response=True,
                    name="/submit",
                ) as response:

                    if response.status_code != 200:
                        response.failure(
                            f"Submit failed. status={response.status_code}, body={response.text}"
                        )
                        return

                    try:
                        data = response.json()
                    except Exception as e:
                        response.failure(
                            f"Submit returned non-JSON response: {e}, body={response.text}"
                        )
                        return

                    job_id = data.get("job_id")

                    if not job_id:
                        response.failure(f"No job_id in submit response: {data}")
                        return

                    response.success()

        except FileNotFoundError:
            print(f"[FATAL] File not found: {pdf_path}")
            return
        except Exception as e:
            print(f"[FATAL] Unexpected submit error: {e}")
            return

        max_polls = 120
        poll_interval_sec = 1

        for _ in range(max_polls):
            with self.client.get(
                f"/status/{job_id}",
                catch_response=True,
                name="/status",
            ) as status_resp:

                if status_resp.status_code != 200:
                    status_resp.failure(
                        f"Status failed. status={status_resp.status_code}, body={status_resp.text}"
                    )
                    return

                try:
                    status_data = status_resp.json()
                except Exception as e:
                    status_resp.failure(
                        f"Status returned non-JSON response: {e}, body={status_resp.text}"
                    )
                    return

                status = status_data.get("status")

                if status in ("completed", "completed_with_warning"):
                    status_resp.success()
                    return

                if status == "failed":
                    status_resp.failure(
                        f"Job failed: {status_data.get('error')}"
                    )
                    return

                if status in ("queued", "processing"):
                    status_resp.success()
                else:
                    status_resp.failure(
                        f"Unexpected job status: {status_data}"
                    )
                    return

            time.sleep(poll_interval_sec)

        print(f"[WARN] Job {job_id} did not finish within polling window")
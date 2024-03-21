import asyncio
import argparse

from proximl import ProxiML

parser = argparse.ArgumentParser(
    description="Large Language Model (LLM) Endpoint Example"
)

parser.add_argument(
    "--model",
    choices=[
        "Llama-2-7b-chat-hf",
        "Llama-2-13b-chat-hf",
        "Llama-2-70b-chat-hf",
        "falcon-7b-instruct",
        "Mixtral-8x7B-Instruct-v0.1",
    ],
    default="Llama-2-13b-chat-hf",
    help="The language model checkpoint to use",
)
parser.add_argument(
    "--gpu-count",
    type=int,
    default=2,
    help="Number of GPUs to attach (max 4)",
)


async def create_endpoint(proximl, model, gpu_count):
    if model == "Mixtral-8x7B-Instruct-v0.1":
        max_tokens = 32768
    elif model.startswith("Llama-2"):
        max_tokens = 4096
    else:
        max_tokens = 2048
    max_input_tokens = int(max_tokens * 0.75)
    job = await proximl.jobs.create(
        f"LLM Endpoint Example - {model}",
        type="endpoint",
        gpu_types=["rtx3090"],
        gpu_count=gpu_count,
        disk_size=30,
        endpoint=dict(
            start_command=f"--model-id /opt/ml/checkpoint --quantize bitsandbytes-nf4 --trust-remote-code --port 80 --json-output --hostname 0.0.0.0 --max-input-length {max_input_tokens} --max-total-tokens {max_tokens} --max-batch-prefill-tokens {max_tokens}"
        ),
        model=dict(
            checkpoints=[dict(id=model, public=True)],
        ),
        environment=dict(
            type="CUSTOM",
            custom_image="ghcr.io/huggingface/text-generation-inference:1.4",
        ),
    )
    return job


if __name__ == "__main__":
    args = parser.parse_args()
    proximl = ProxiML()
    job = asyncio.run(
        create_endpoint(
            proximl,
            args.model,
            args.gpu_count,
        )
    )
    print("Created Endpoint: ", job.id, " Waiting to Start...")
    asyncio.run(job.wait_for("running"))
    print("Job ID: ", job.id, " Running")
    print("URL", job.url)

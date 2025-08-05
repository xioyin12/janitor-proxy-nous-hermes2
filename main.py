from flask import Flask, request, jsonify
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import os

app = Flask(__name__)
model_path = hf_hub_download(
    "NousResearch/Nous-Hermes-2-Mistral-7B-DPO-GGUF",
    filename="Nous-Hermes-2-Mistral-7B-DPO.Q4_0.gguf"
)
llm = Llama(model_path=model_path, n_ctx=2048, n_threads=4, gpu_layers=0)

@app.route("/v1/chat/completions", methods=["POST"])
def chat():
    data = request.get_json()
    prompt = data["messages"][-1]["content"]
    out = llm(prompt=prompt, max_tokens=512, temperature=0.7)
    return jsonify({"choices":[{"message":{"content":out['choices'][0]['text']}}]})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)

import os
import argparse
import torch
import flask
import transformers

# LLM of the Reasoning and Execution module: LLaMA-7B (base model, no chat template),
# greedy decoding with at most 256 output tokens.
parser = argparse.ArgumentParser(description="LLM Server")
parser.add_argument("--model", type=str, default=os.environ.get("LILO_LLM", "huggyllama/llama-7b"))
parser.add_argument("--max_new_tokens", type=int, default=256)
parser.add_argument("--port", type=int, default=5000)
args = parser.parse_args()

tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
model = transformers.AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()

def llm_inference(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False, # greedy decoding
            pad_token_id=tokenizer.eos_token_id,
        )
    # return only the generated continuation
    return tokenizer.decode(outputs[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

app = flask.Flask("LLM Server")
@app.route("/predict", methods=["POST"])
def predict():
    # Initialize the data dictionary that will be returned from the view.
    data = {"success": False}

    # Ensure data was properly uploaded to our endpoint.
    if flask.request.method == 'POST':
        # Parse the JSON input
        llm_input = flask.request.get_json()
        if llm_input:
            prompt = llm_input.get("prompt")

            prediction = llm_inference(prompt)
            data['prediction'] = prediction
            print(prediction)

            # Indicate that the request was a success.
            data["success"] = True

    # Return the data dictionary as a JSON response.
    return flask.jsonify(data)

if __name__ == '__main__':
    app.run(port=args.port)

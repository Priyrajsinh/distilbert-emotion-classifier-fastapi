"""Gradio demo for the Distilbert Emotion Classifier.

Calls the running FastAPI at API_URL and displays label, confidence,
and sorted probability breakdown.
"""

from __future__ import annotations

import gradio as gr
import httpx

from src.logger import get_logger

logger = get_logger(__name__)

API_URL = "http://localhost:8000/api/v1/predict"


def predict_text(text: str) -> tuple[str, str, str]:
    """Send text to the FastAPI endpoint and return formatted results."""
    try:
        r = httpx.post(API_URL, json={"text": text}, timeout=10.0)
        r.raise_for_status()
        data = r.json()
        label = data["label"]
        confidence = f"{data['confidence']:.1%}"
        probs = sorted(data["probabilities"].items(), key=lambda x: x[1], reverse=True)
        prob_text = "\n".join(f"{k}: {v:.1%}" for k, v in probs)
        return label, confidence, prob_text
    except httpx.ConnectError:
        logger.error("Cannot connect to API at %s", API_URL)
        return "error", "0.0%", "Cannot connect to API. Is the server running?"
    except Exception as exc:
        logger.error("Prediction request failed: %s", exc)
        return "error", "0.0%", f"Request failed: {exc}"


_EXAMPLES = [
    ["I am so happy today, everything is going great!"],
    ["This is absolutely terrible and I hate it."],
    ["I'm not sure what to think about this situation."],
    ["That was completely unexpected, I didn't see it coming!"],
]

with gr.Blocks(title="Emotion Classifier") as demo:
    gr.Markdown("## Distilbert Emotion Classifier\nEnter text to predict its emotion.")

    text_input = gr.Textbox(label="Input Text", placeholder="Type something...")
    submit_btn = gr.Button("Predict")

    with gr.Row():
        label_out = gr.Textbox(label="Predicted Label")
        conf_out = gr.Textbox(label="Confidence")

    probs_out = gr.Textbox(label="All Probabilities", lines=7)

    submit_btn.click(
        fn=predict_text,
        inputs=text_input,
        outputs=[label_out, conf_out, probs_out],
    )

    gr.Examples(examples=_EXAMPLES, inputs=text_input)

if __name__ == "__main__":
    demo.launch(share=True)

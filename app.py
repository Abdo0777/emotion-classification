import gradio as gr
import matplotlib.pyplot as plt
from transformers import pipeline

MODEL_PATH = "distilbert_emotion_final"

emotion_pipe = pipeline(
    "text-classification",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
    top_k=None
)

EMOTION_COLORS = {
    "joy": "#FFC107",
    "sadness": "#2196F3",
    "anger": "#F44336",
    "fear": "#9C27B0",
    "surprise": "#FF9800",
    "disgust": "#4CAF50",
}

EMOTION_EMOJIS = {
    "joy": "😊",
    "sadness": "😢",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
    "disgust": "🤢",
}

def predict_emotion(text):
    if not text or not text.strip():
        return "—", None, ""

    results = emotion_pipe(text)[0]
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    labels = [r["label"] for r in results]
    scores = [r["score"] for r in results]
    colors = [EMOTION_COLORS.get(l, "#999999") for l in labels]

    top_label = labels[0]
    top_score = scores[0]
    top_html = f"""
    <div style='text-align:center; padding: 18px; border-radius: 16px;
                background: linear-gradient(135deg, {colors[0]}22, {colors[0]}11);
                border: 2px solid {colors[0]};'>
        <div style='font-size: 46px;'>{EMOTION_EMOJIS.get(top_label,"")}</div>
        <div style='font-size: 26px; font-weight: 700; color: {colors[0]};
                     text-transform: capitalize; margin-top: 4px;'>{top_label}</div>
        <div style='font-size: 15px; color: #666; margin-top: 2px;'>
            {top_score*100:.1f}% confidence</div>
    </div>
    """

    fig, ax = plt.subplots(figsize=(6, 3.2))
    bars = ax.barh(labels[::-1], scores[::-1], color=colors[::-1])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_title("Emotion Confidence Breakdown", fontsize=13, fontweight="bold")
    for bar, score in zip(bars, scores[::-1]):
        ax.text(min(score + 0.02, 0.93), bar.get_y() + bar.get_height()/2,
                f"{score*100:.1f}%", va="center", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    return top_html, fig, ""

theme = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="pink",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Poppins"), "sans-serif"],
)

custom_css = """
#title { text-align: center; font-weight: 800; font-size: 30px;
         background: linear-gradient(90deg, #7C3AED, #EC4899);
         -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
#subtitle { text-align: center; color: #6B7280; margin-bottom: 10px; }
.gradio-container { max-width: 900px !important; margin: auto !important; }
"""

with gr.Blocks(theme=theme, css=custom_css, title="Emotion Classifier") as demo:
    gr.HTML("<div id='title'>💜 Emotion Classifier</div>")
    gr.HTML("<div id='subtitle'>Powered by fine-tuned DistilBERT · 6-class emotion detection</div>")

    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                lines=4,
                placeholder="Type how you feel, or paste any sentence...",
                label="Your text",
            )
            submit_btn = gr.Button("Analyze Emotion", variant="primary")
            gr.Examples(
                examples=[
                    "I can't believe how amazing this is!",
                    "I'm so scared right now, I don't know what to do.",
                    "This is disgusting, I can't even look at it.",
                    "Wow, I never expected that to happen!",
                ],
                inputs=text_input,
            )
        with gr.Column(scale=1):
            top_result = gr.HTML(label="Dominant Emotion")
            chart_output = gr.Plot(label="Confidence Breakdown")

    error_box = gr.Textbox(visible=False)

    submit_btn.click(predict_emotion, inputs=text_input, outputs=[top_result, chart_output, error_box])
    text_input.submit(predict_emotion, inputs=text_input, outputs=[top_result, chart_output, error_box])

if __name__ == "__main__":
    demo.launch()
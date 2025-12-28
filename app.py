from flask import Flask, render_template, request
from extractive import extractive_summary
from abstractive import abstractive_summary
from chunker import chunk_text
from functools import lru_cache

app = Flask(__name__)

@lru_cache(maxsize=64)
def summarize_cached(text, mode):
    if mode == "extractive":
        return extractive_summary(text)
    else:
        summaries = []
        for chunk in chunk_text(text):
            summaries.append(abstractive_summary(chunk))
        return " ".join(summaries)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    rawtext = request.form["rawtext"]

    MAX_WORDS = 1500
    word_count = len(rawtext.split())

    if word_count > MAX_WORDS:
        return render_template(
            "summary.html",
            summary="Input too long. Please limit text to 1500 words.",
            og=rawtext[:2000] + "...",
            len_og=word_count,
            len_summary=0,
            mode="error"
        )
    
    mode = request.form["mode"]

    summary = summarize_cached(rawtext, mode)

    return render_template(
        "summary.html",
        summary=summary,
        og=rawtext,
        len_og=len(rawtext.split()),
        len_summary=len(summary.split()),
        mode=mode
    )

if __name__ == "__main__":
    app.run()

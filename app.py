import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="🤗 Hugging Face NLP Playground", layout="wide")
st.title("🤖 Hugging Face NLP Tasks with Streamlit")

# Sidebar selection
task = st.sidebar.selectbox(
    "Choose an NLP task:",
    [
        "Sentiment Analysis",
        "Named Entity Recognition",
        "Question Answering",
        "Summarization",
        "Translation",
        "Text Generation",
        "Fill-Mask",
        "Zero-Shot Classification",
    ],
)


# Cached model loading using smaller models for fast startup
@st.cache_resource
def load_pipeline(task_name):
    model_map = {
        "sentiment-analysis": "distilbert-base-uncased-finetuned-sst-2-english",
        "ner": "dslim/bert-base-NER",
        "question-answering": "distilbert-base-cased-distilled-squad",
        "summarization": "sshleifer/distilbart-cnn-12-6",
        "translation_en_to_fr": "Helsinki-NLP/opus-mt-en-fr",
        "text-generation": "distilgpt2",
        "fill-mask": "bert-base-uncased",
        "zero-shot-classification": "facebook/bart-large-mnli",
    }
    return pipeline(task_name, model=model_map[task_name])


# UI and logic per task
if task == "Sentiment Analysis":
    text = st.text_area("Enter text for sentiment analysis:")
    if text:
        classifier = load_pipeline("sentiment-analysis")
        st.json(classifier(text))

elif task == "Named Entity Recognition":
    text = st.text_area("Enter text for NER:")
    if text:
        ner = load_pipeline("ner")
        st.json(ner(text))

elif task == "Question Answering":
    context = st.text_area("Enter context:")
    question = st.text_input("Enter question:")
    if context and question:
        qa = load_pipeline("question-answering")
        st.json(qa(question=question, context=context))

elif task == "Summarization":
    text = st.text_area("Enter long text for summarization:")
    if text:
        summarizer = load_pipeline("summarization")
        st.json(summarizer(text, max_length=130, min_length=30, do_sample=False))

elif task == "Translation":
    text = st.text_area("Enter English text for translation to French:")
    if text:
        translator = load_pipeline("translation_en_to_fr")
        st.json(translator(text))

elif task == "Text Generation":
    prompt = st.text_area("Enter a prompt for text generation:")
    if prompt:
        generator = load_pipeline("text-generation")
        st.json(generator(prompt, max_length=50, num_return_sequences=1))

elif task == "Fill-Mask":
    masked_text = st.text_input("Enter a sentence with a [MASK] token (e.g. 'The sky is [MASK].'):")
    if masked_text:
        fill_mask = load_pipeline("fill-mask")
        st.json(fill_mask(masked_text))

elif task == "Zero-Shot Classification":
    text = st.text_area("Enter the text:")
    labels = st.text_input("Enter candidate labels (comma-separated):")
    if text and labels:
        candidate_labels = [l.strip() for l in labels.split(",")]
        zero_shot = load_pipeline("zero-shot-classification")
        st.json(zero_shot(text, candidate_labels=candidate_labels))

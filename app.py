import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Hugging Face NLP Playground", layout="wide")
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
        "Zero-Shot Classification"
    ]
)

# Initialize pipelines only when needed
@st.cache_resource
def load_pipeline(task_name, **kwargs):
    return pipeline(task_name, **kwargs)

# Define UI and logic per task
if task == "Sentiment Analysis":
    text = st.text_area("Enter text for sentiment analysis:")
    if text:
        classifier = load_pipeline("sentiment-analysis")
        result = classifier(text)
        st.json(result)

elif task == "Named Entity Recognition":
    text = st.text_area("Enter text for NER:")
    if text:
        ner = load_pipeline("ner", grouped_entities=True)
        result = ner(text)
        st.json(result)

elif task == "Question Answering":
    context = st.text_area("Enter context:")
    question = st.text_input("Enter question:")
    if context and question:
        qa = load_pipeline("question-answering")
        result = qa(question=question, context=context)
        st.json(result)

elif task == "Summarization":
    text = st.text_area("Enter long text for summarization:")
    if text:
        summarizer = load_pipeline("summarization")
        result = summarizer(text, max_length=130, min_length=30, do_sample=False)
        st.json(result)

elif task == "Translation":
    text = st.text_area("Enter English text for translation to French:")
    if text:
        translator = load_pipeline("translation_en_to_fr")
        result = translator(text)
        st.json(result)

elif task == "Text Generation":
    prompt = st.text_area("Enter a prompt for text generation:")
    if prompt:
        generator = load_pipeline("text-generation")
        result = generator(prompt, max_length=100, num_return_sequences=1)
        st.json(result)

elif task == "Fill-Mask":
    masked_text = st.text_input("Enter a sentence with a [MASK] token:")
    if masked_text:
        fill_mask = load_pipeline("fill-mask")
        result = fill_mask(masked_text)
        st.json(result)

elif task == "Zero-Shot Classification":
    text = st.text_area("Enter the text:")
    labels = st.text_input("Enter candidate labels separated by commas (e.g., finance, health, education):")
    if text and labels:
        labels = [label.strip() for label in labels.split(",")]
        zero_shot = load_pipeline("zero-shot-classification")
        result = zero_shot(text, candidate_labels=labels)
        st.json(result)


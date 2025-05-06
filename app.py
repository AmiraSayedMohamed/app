# -*- coding: utf-8 -*-
# MUST BE FIRST - SQLite fix for ChromaDB
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Core libraries (keep your existing imports)
import os
import re
import time
import base64
import numpy as np
from dotenv import load_dotenv
import colorsys
import importlib.metadata

# Audio processing (keep your existing imports)
from pydub import AudioSegment
from audio_recorder_streamlit import audio_recorder

# YouTube/Video processing
import yt_dlp

# OpenAI
from openai import OpenAI

# Visualization
import graphviz

# Streamlit
import streamlit as st

# Updated LangChain Imports
import chromadb
from langchain_chroma import Chroma  # Changed from langchain_community.vectorstores
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
import langgraph
print("langgraph version:", importlib.metadata.version("langgraph"))
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Additional required components (keep your existing imports)
from typing import List, Dict, Optional
import tempfile
from chromadb.config import Settings
import torch
torch.set_default_device('cpu')  # Force PyTorch to use CPU

# Constants (unchanged)
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MAX_AUDIO_SIZE_BYTES = 25 * 1024 * 1024
AUDIO_PART_DURATION_MS = 5 * 60 * 1000
COLLECTION_NAME = "video_transcripts"

# Environment setup (unchanged)
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# Initialize OpenAI client (unchanged)
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")
openai = OpenAI(api_key=OPENAI_API_KEY)

# Set up in-memory ChromaDB settings
# chroma_settings = Settings(is_persistent=False)

# Initialize vector store (updated)
# embedding_function = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-mpnet-base-v2",  # Explicitly specify model_name
# )
# Initialize the embedding function using OpenAI
chroma_settings = Settings(
    is_persistent=False,
    anonymized_telemetry=False  # Optional: disable telemetry for cleaner logs
)
chroma_client = chromadb.EphemeralClient(settings=chroma_settings)
embedding_function = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vector_store = Chroma(
    client=chroma_client,  # Pass the EphemeralClient explicitly
    embedding_function=embedding_function,
    collection_name=COLLECTION_NAME
)
# Unified preprocessing function
def preprocess_youtube_video(url):
    """Handles common preprocessing steps for all learning styles."""
    try:
        clean_url = clean_youtube_url(url)
        st.session_state['status'] = f"✅ Cleaned URL: {clean_url}"
        audio_path = download_audio_from_youtube(clean_url)
        split_files = split_audio(audio_path)
        st.session_state['status'] = f"✅ Split into {len(split_files)} audio parts"
        all_text = []
        for part_file in split_files:
            text = transcribe_audio_openai(part_file)
            all_text.append(text)
            os.remove(part_file)
        transcript_text = "\n".join(all_text)
        cleaned_text = clean_transcript(transcript_text)
        st.session_state['status'] = "✅ Transcript cleaned"
        return cleaned_text, audio_path
    except Exception as e:
        st.error(f"❌ Preprocessing error: {e}")
        raise

# Clean YouTube URL
def clean_youtube_url(url):
    if "youtu.be" in url:
        video_id = url.split("/")[-1].split("?")[0]
        return f"https://www.youtube.com/watch?v={video_id}"
    return url.split("&")[0]

# Download YouTube Audio
def download_audio_from_youtube(url, output_path='./downloads'):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_path}/%(title)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': False,
    }
    os.makedirs(output_path, exist_ok=True)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info).rsplit('.', 1)[0] + '.wav'
    st.session_state['status'] = f"✅ Audio downloaded at: {filename}"
    return filename

# Split Audio into Parts
def split_audio(audio_path):
    audio = AudioSegment.from_file(audio_path)
    total_duration_ms = len(audio)
    parts = []
    start = 0
    while start < total_duration_ms:
        end = min(start + AUDIO_PART_DURATION_MS, total_duration_ms)
        part = audio[start:end]
        while len(part.export(format="wav").read()) > MAX_AUDIO_SIZE_BYTES and AUDIO_PART_DURATION_MS > 30 * 1000:
            part_duration_ms = AUDIO_PART_DURATION_MS // 2
            end = min(start + part_duration_ms, total_duration_ms)
            part = audio[start:end]
        filename = f"./downloads/part_{len(parts)}.wav"
        part.export(filename, format="wav")
        parts.append(filename)
        start = end
    return parts

# Transcribe Audio using OpenAI Whisper
def transcribe_audio_openai(audio_path):
    with open(audio_path, "rb") as audio_file:
        transcript = openai.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1"
        )
    st.session_state['status'] = f"✅ Transcribed: {audio_path}"
    return transcript.text

# Clean Transcript Text
def clean_transcript(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = text.replace('. ', '.\n').strip()
    return text

# Initialize Chat Model
def init_chat_model(model, model_provider):
    if model_provider == 'openai':
        return ChatOpenAI(model=model, api_key=OPENAI_API_KEY)
    else:
        st.error("Unsupported model provider")
        return None

# Create ReAct Agent
# def create_react_agent(llm, tools, checkpointer):
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", "You are a helpful AI assistant. Use the provided tools to answer questions accurately."),
#         ("human", "{input}"),
#         ("assistant", "{output}")
#     ])
#     tool_executor = ToolExecutor(tools)
#     graph = StateGraph({"messages": []})
#     graph.add_node("agent", lambda state: {"messages": [llm.invoke(prompt.format(input=state["messages"][-1]["content"]))]})
#     graph.add_node("tool", lambda state: {"messages": [tool_executor.invoke(state["messages"][-1])]})
#     graph.add_edge("agent", "tool")
#     graph.add_edge("tool", END)
#     graph.set_entry_point("agent")
#     return graph.compile(checkpointer=checkpointer)

# Setup RAG components
# def setup_rag(cleaned_text):
#     chunks = split_text_into_chunks(cleaned_text)
#     store_chunks_in_vectorstore(chunks)
#     llm = init_chat_model(model='gpt-4', model_provider='openai')
#     memory = MemorySaver()
#     rag_agent = create_react_agent(llm, [retrieve], checkpointer=memory)
#     config = {"configurable": {"thread_id": "session_01"}}
#     return rag_agent, config

def setup_rag(cleaned_text):
    chunks = split_text_into_chunks(cleaned_text)
    store_chunks_in_vectorstore(chunks)
    llm = init_chat_model(model='gpt-4', model_provider='openai')
    memory = MemorySaver()
    rag_agent = create_react_agent(llm, [retrieve], checkpointer=memory)  # Updated to use prebuilt function
    config = {"configurable": {"thread_id": "session_01"}}
    return rag_agent, config
    
# Split Text into Chunks
def split_text_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return text_splitter.split_text(text)

# Store Chunks in Vector Store
def store_chunks_in_vectorstore(chunks):
    documents = [Document(page_content=chunk) for chunk in chunks]
    vector_store.add_documents(documents=documents)
    st.session_state['status'] = f"✅ {len(chunks)} chunks stored in vector database."

# Retriever Tool
@tool(response_format='content_and_artifact')
def retrieve(query: str):
    """Retrieve relevant content chunks from the vector store based on the user query."""
    retrieved_docs = vector_store.similarity_search(query, k=3)
    if not retrieved_docs:
        all_docs = vector_store.similarity_search("summary of the video", k=3)
        serialized = "Summary of the video transcript:\n\n" + "\n\n".join((f"Content: {doc.page_content}") for doc in all_docs)
        return serialized, all_docs
    serialized = "\n\n".join((f"Content: {doc.page_content}") for doc in retrieved_docs)
    return serialized, retrieved_docs

# Enhanced Assistant Response
def get_assistant_response(user_text, rag_agent, config, transcript_text):
    system_prompt = """
You are a helpful AI tutor. Answer questions strictly based on the provided YouTube video transcript. 
If the question is unrelated to the transcript, respond only with: 
"Sorry, I can only answer questions related to the video content. Please ask something about the video."
Use the retrieved chunks or transcript summary to provide accurate answers.
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Transcript summary:\n{transcript_text[:1000]}...\n\nUser question: {user_text}"}
    ]
    try:
        response = rag_agent.invoke({"messages": messages}, config=config)
        answer = response['messages'][-1].content
        if answer == "Sorry, I can only answer questions related to the video content. Please ask something about the video.":
            return answer
        answer_embedding = embedding_function.embed_query(answer)
        transcript_chunks = split_text_into_chunks(transcript_text)
        transcript_embeddings = embedding_function.embed_documents(transcript_chunks)
        similarities = [
            np.dot(answer_embedding, chunk_embedding) /
            (np.linalg.norm(answer_embedding) * np.linalg.norm(chunk_embedding))
            for chunk_embedding in transcript_embeddings
        ]
        similarity_threshold = 0.7
        if max(similarities, default=0) < similarity_threshold:
            return "Sorry, I can only answer questions related to the video content. Please ask something about the video."
        return answer
    except Exception as e:
        st.error(f"❌ RAG error: {e}")
        return "Sorry, I couldn't process your request. Please try again."

# Visual Learning (Mind Map)
def process_visual_learning(cleaned_text):
    mindmap_text = extract_mindmap(cleaned_text)
    image_path = generate_mindmap_image(mindmap_text)
    st.image(image_path, caption="Generated Mind Map", use_column_width=True)
    st.success("✅ Mind map has been created!")

def build_mindmap_prompt(text):
    prompt = f"""
You are an expert in summarization and mind mapping.
Here is a cleaned transcription of a YouTube educational video:
{text}
Your task:
1. Identify the overall MAIN TOPIC (e.g., Machine Learning).
2. Under it, extract the **main branches** (e.g., Supervised Learning, Unsupervised Learning).
3. For each branch, list the **sub-ideas** and details underneath it.
⚠️ IMPORTANT: Follow this exact hierarchy:
- [MAIN TOPIC]
    - Main Branch 1
        - Sub-idea 1
        - Sub-idea 2
    - Main Branch 2
        - Sub-idea 1
        - Sub-idea 2
Be concise but cover all key concepts.
⚠️ Output ONLY the structured list. No introduction, no explanation, no comments.
"""
    return prompt

def extract_mindmap(text):
    prompt = build_mindmap_prompt(text)
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant specialized in summarization."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content

def parse_outline_to_edges(outline_text):
    lines = outline_text.strip().split('\n')
    edges = []
    parent_stack = []
    root = None
    for line in lines:
        match = re.match(r'^(\s*)-?\s*(.+)$', line)
        if match:
            indent = len(match.group(1))
            content = match.group(2).strip()
            if root is None:
                root = content
                parent_stack.append((indent, root))
                continue
            while parent_stack and indent <= parent_stack[-1][0]:
                parent_stack.pop()
            if parent_stack:
                parent = parent_stack[-1][1]
                edges.append((parent, content))
            parent_stack.append((indent, content))
    return edges, root

def generate_distinct_pastel_colors(n):
    colors = []
    for i in range(n):
        hue = i / max(n, 1)
        lightness = 0.85
        saturation = 0.4
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)
        )
        colors.append(hex_color)
    return colors

def generate_mindmap_image(outline_text, output_filename="mindmap_generated"):
    dot = graphviz.Digraph(comment="Mind Map", engine="twopi")
    dot.graph_attr.update(
        splines="curved",
        nodesep="1.0",
        ranksep="2.0",
        overlap="false"
    )
    edges, root = parse_outline_to_edges(outline_text)
    node_hierarchy = {}
    for parent, child in edges:
        if parent not in node_hierarchy:
            node_hierarchy[parent] = []
        node_hierarchy[parent].append(child)
    color_map = {root: "#F7E1A1"}
    top_level_nodes = node_hierarchy.get(root, [])
    pastel_colors = generate_distinct_pastel_colors(len(top_level_nodes))
    for node, color in zip(top_level_nodes, pastel_colors):
        color_map[node] = color
    def propagate_colors(node, parent_color):
        if node not in color_map:
            color_map[node] = parent_color
        if node in node_hierarchy:
            for child in node_hierarchy[node]:
                propagate_colors(child, color_map[node])
    for node in top_level_nodes:
        propagate_colors(node, color_map[node])
    for parent, child in edges:
        parent_color = color_map.get(parent, "#D3D3D3")
        child_color = color_map.get(child, parent_color)
        dot.node(parent, shape="box", style="filled,setlinewidth(2)", fillcolor=parent_color, fontcolor="black", fontsize="12")
        dot.node(child, shape="box", style="filled,setlinewidth(2)", fillcolor=child_color, fontcolor="black", fontsize="12")
        dot.edge(parent, child, color=parent_color)
    dot.render(output_filename, format='png', cleanup=True)
    return output_filename + ".png"

# Kinesthetic Learning (Quiz)
def process_kinesthetic_learning(cleaned_text):
    quiz_text = generate_quiz(cleaned_text)
    run_interactive_quiz(quiz_text)

def build_quiz_prompt(text):
    return f"""
You are an expert educator.
Based on the transcript below, generate exactly 5 multiple-choice quiz questions:
Transcript:
{text[:2000]}...
Requirements:
- Each question must have exactly 4 answer choices labeled A, B, C, D.
- Write the correct answer like: Correct: B
- Return plain text in the format:
Question?
A) Option A
B) Option B
C) Option C
D) Option D
Correct: X
"""

def generate_quiz(text):
    prompt = build_quiz_prompt(text)
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a quiz creator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
    )
    return response.choices[0].message.content

def run_interactive_quiz(quiz_text):
    st.write("\n🧠 Starting interactive quiz!")
    questions = quiz_text.strip().split('\n\n')
    score = 0
    user_answers = []
    for i, q in enumerate(questions):
        lines = q.strip().split('\n')
        question = lines[0]
        choices = lines[1:5]
        correct_line = [l for l in lines if l.lower().startswith("correct")]
        if not correct_line:
            st.warning("⚠️ Skipping malformed question.")
            continue
        correct_answer = correct_line[0].split(":")[1].strip().upper()
        st.write(f"\nQ{i+1}: {question}")
        for choice in choices:
            st.write(choice)
        user_answer = st.radio(f"Select your answer for Q{i+1}:", 
                             [choice[0] for choice in choices], 
                             key=f"q{i}")
        user_answers.append((question, choices, user_answer, correct_answer))
    st.write("\n🎯 Quiz Results:")
    for i, (question, choices, user_answer, correct_answer) in enumerate(user_answers):
        st.write(f"\nQ{i+1}: {question}")
        for choice in choices:
            st.write(choice)
        if user_answer == correct_answer:
            st.success(f"✅ Your answer: {user_answer} (Correct!)")
            score += 1
        else:
            st.error(f"❌ Your answer: {user_answer} (Correct answer was: {correct_answer})")
    st.write(f"\nFinal Score: {score}/{len(user_answers)}")

# Auditory Learning
def process_auditory_learning(rag_agent, config, transcript_text):
    st.write("\n🎧 Auditory Learning Mode ON!")
    audio_bytes = audio_recorder(pause_threshold=10.0)
    if audio_bytes:
        audio_path = "user_input.wav"
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)
        user_text = transcribe_user_audio(audio_path)
        if user_text:
            st.write(f"\n🗣️ You said: {user_text}")
            assistant_text = get_assistant_response(user_text, rag_agent, config, transcript_text)
            st.write(f"\n🤖 AI said: {assistant_text}")
            audio_response_path = speak_response(assistant_text)
            if audio_response_path:
                st.audio(audio_response_path)

def transcribe_user_audio(audio_path):
    try:
        with open(audio_path, "rb") as audio_file:
            transcript = openai.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                language="en"
            )
        return transcript.text
    except Exception as e:
        st.error(f"❌ Transcription error: {e}")
        return None

def speak_response(text, filename="reply.mp3"):
    try:
        speech = openai.audio.speech.create(model="tts-1", voice="nova", input=text)
        with open(filename, "wb") as f:
            f.write(speech.content)
        return filename
    except Exception as e:
        st.error(f"❌ Speech generation error: {e}")
        return None

# Reading Learning
def process_reading_learning(rag_agent, config, transcript_text):
    st.write("✅ Bot ready! Ask questions about the video.")
    user_input = st.text_input("📝 Your question (or 'exit' to quit):", key="reading_input")
    if user_input and user_input.lower() != 'exit':
        response = get_assistant_response(user_input, rag_agent, config, transcript_text)
        st.write(f"\n🤖 Response: {response}")

# Main Streamlit App
def main():
    st.title("🎓 SmarterLearn - Adaptive Learning Assistant")
    if 'status' not in st.session_state:
        st.session_state['status'] = ""
    st.sidebar.title("Settings")
    learning_style = st.sidebar.selectbox(
        "Choose your learning style:",
        ["visual", "reading", "kinesthetic", "auditory"],
        index=0
    )
    youtube_url = st.sidebar.text_input("Enter YouTube video URL:")
    if st.sidebar.button("Start Learning"):
        if not youtube_url:
            st.error("Please enter a YouTube URL")
            return
        with st.spinner("⏳ Preparing content..."):
            try:
                cleaned_text, audio_path = preprocess_youtube_video(youtube_url)
                st.session_state['cleaned_text'] = cleaned_text
                st.session_state['audio_path'] = audio_path
                if learning_style in ["reading", "auditory"]:
                    rag_agent, config = setup_rag(cleaned_text)
                    st.session_state['rag_agent'] = rag_agent
                    st.session_state['config'] = config
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.error("ℹ️ Check API keys, FFmpeg, Graphviz, network, or microphone.")
                return
    if 'cleaned_text' in st.session_state:
        st.write(st.session_state['status'])
        if learning_style == "visual":
            process_visual_learning(st.session_state['cleaned_text'])
        elif learning_style == "kinesthetic":
            process_kinesthetic_learning(st.session_state['cleaned_text'])
        elif learning_style == "reading":
            process_reading_learning(
                st.session_state['rag_agent'],
                st.session_state['config'],
                st.session_state['cleaned_text']
            )
        elif learning_style == "auditory":
            process_auditory_learning(
                st.session_state['rag_agent'],
                st.session_state['config'],
                st.session_state['cleaned_text']
            )

if __name__ == "__main__":
    main()

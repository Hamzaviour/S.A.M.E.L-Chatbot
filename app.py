import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import streamlit as st
import pandas as pd
import numpy as np
import nltk
import spacy
import re
import requests
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import plotly.express as px
import plotly.graph_objects as go

try:
    import os
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except (ImportError, Exception) as e:
    print(f"Warning: sentence-transformers not available: {e}")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    pass

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    st.error("spaCy English model not found. Please run: python -m spacy download en_core_web_sm")
    st.stop()

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

TECHNICAL_FAQS = [
    {"question": "What is machine learning?", "answer": "Machine Learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed. It uses algorithms to identify patterns in data and make predictions or decisions.", "category": "AI/ML", "keywords": ["machine learning", "ML", "AI", "algorithms", "data"]},
    {"question": "How does cloud computing work?", "answer": "Cloud computing delivers computing services over the internet, including servers, storage, databases, networking, software, and analytics. Instead of owning physical infrastructure, users access these services on-demand from cloud providers like AWS, Azure, or Google Cloud.", "category": "Cloud", "keywords": ["cloud computing", "AWS", "Azure", "servers", "infrastructure"]},
    {"question": "What is the difference between HTTP and HTTPS?", "answer": "HTTP (HyperText Transfer Protocol) transmits data in plain text, while HTTPS (HTTP Secure) encrypts data using SSL/TLS certificates. HTTPS provides security, data integrity, and authentication, making it essential for websites handling sensitive information.", "category": "Web Security", "keywords": ["HTTP", "HTTPS", "SSL", "TLS", "security", "encryption"]},
    {"question": "What is API and how does it work?", "answer": "API (Application Programming Interface) is a set of protocols and tools that allows different software applications to communicate with each other. It defines methods for requesting and exchanging data between systems, enabling integration and functionality sharing.", "category": "Software Development", "keywords": ["API", "REST", "integration", "endpoints", "software"]},
    {"question": "What is database normalization?", "answer": "Database normalization is the process of organizing data in a database to reduce redundancy and improve data integrity. It involves dividing large tables into smaller, related tables and defining relationships between them through primary and foreign keys.", "category": "Database", "keywords": ["database", "normalization", "SQL", "tables", "data integrity"]},
    {"question": "How does version control with Git work?", "answer": "Git is a distributed version control system that tracks changes in files and coordinates work among multiple developers. It maintains a complete history of changes, allows branching and merging, and enables collaboration through repositories hosted on platforms like GitHub.", "category": "Development Tools", "keywords": ["git", "version control", "GitHub", "repository", "collaboration"]},
    {"question": "What is containerization and Docker?", "answer": "Containerization packages applications and their dependencies into containers - lightweight, portable units that can run consistently across different environments. Docker is a popular platform that creates, deploys, and manages containers, ensuring applications work the same way everywhere.", "category": "DevOps", "keywords": ["Docker", "containers", "containerization", "deployment", "DevOps"]},
    {"question": "What is cybersecurity and why is it important?", "answer": "Cybersecurity protects digital systems, networks, and data from cyber threats like malware, phishing, and hacking. It's crucial for protecting sensitive information, maintaining business continuity, ensuring privacy, and preventing financial losses from cyber attacks.", "category": "Security", "keywords": ["cybersecurity", "security", "malware", "phishing", "hacking"]},
    {"question": "Who created you?", "answer": "I was created by Hamza Younas (Machine Learning Engineer) using advanced deep learning, RAG (Retrieval-Augmented Generation), and LLM (Large Language Model) algorithms. I am S.A.M.E.L - Smart Automated Machine for Efficient Learning, designed to be an intelligent FAQ assistant.", "category": "About S.A.M.E.L", "keywords": ["creator", "Hamza Younas", "deep learning", "RAG", "LLM", "S.A.M.E.L"]},
    {"question": "When were you created?", "answer": "I was created by Hamza Younas on September 4th, 2025. This marks the beginning of my journey as an intelligent FAQ assistant powered by cutting-edge AI technologies.", "category": "About S.A.M.E.L", "keywords": ["creation date", "September 2025", "created", "when", "date"]},
    {"question": "What company are you made for?", "answer": "I was specifically created for Apexify Technologies. I am designed to serve as an intelligent FAQ assistant to help with technical and educational questions, representing the innovative AI solutions developed by Apexify Technologies.", "category": "About S.A.M.E.L", "keywords": ["Apexify Technologies", "company", "organization", "made for", "developed"]},
    {"question": "What is S.A.M.E.L?", "answer": "S.A.M.E.L stands for Smart Automated Machine for Efficient Learning. I am an AI-powered FAQ chatbot designed to provide intelligent responses to technical and educational questions. I combine advanced NLP techniques, machine learning algorithms, and retrieval-augmented generation to deliver accurate and helpful answers.", "category": "About S.A.M.E.L", "keywords": ["S.A.M.E.L", "Smart Automated Machine", "Efficient Learning", "chatbot", "AI"]}
]

EDUCATIONAL_FAQS = [
    {"question": "What are effective study techniques?", "answer": "Effective study techniques include active recall (testing yourself), spaced repetition (reviewing material at increasing intervals), the Pomodoro Technique (25-minute focused sessions), creating mind maps, and teaching concepts to others. These methods enhance memory retention and understanding.", "category": "Study Methods", "keywords": ["study techniques", "active recall", "spaced repetition", "Pomodoro", "learning"]},
    {"question": "How can I improve my critical thinking skills?", "answer": "Improve critical thinking by questioning assumptions, analyzing evidence, considering multiple perspectives, practicing logical reasoning, reading diverse sources, engaging in debates, and reflecting on your own thought processes. Regular practice with puzzles and problem-solving exercises also helps.", "category": "Cognitive Skills", "keywords": ["critical thinking", "analysis", "reasoning", "problem solving", "logic"]},
    {"question": "What is the best way to manage time as a student?", "answer": "Effective time management involves creating a schedule, prioritizing tasks using methods like the Eisenhower Matrix, breaking large projects into smaller tasks, eliminating distractions, using time-blocking techniques, and maintaining a healthy work-life balance.", "category": "Time Management", "keywords": ["time management", "schedule", "prioritization", "productivity", "planning"]},
    {"question": "How do I choose the right career path?", "answer": "Choose a career path by assessing your interests, skills, and values, researching different careers, gaining experience through internships or volunteering, networking with professionals, considering job market trends, and seeking guidance from career counselors or mentors.", "category": "Career Guidance", "keywords": ["career choice", "career path", "interests", "skills", "job market"]},
    {"question": "What are the benefits of online learning?", "answer": "Online learning offers flexibility in scheduling, access to diverse courses and instructors worldwide, cost-effectiveness, self-paced learning, development of digital literacy skills, and the ability to balance education with work or family commitments.", "category": "Online Education", "keywords": ["online learning", "e-learning", "flexibility", "digital education", "remote"]},
    {"question": "How can I overcome exam anxiety?", "answer": "Overcome exam anxiety through proper preparation, regular practice tests, relaxation techniques like deep breathing, maintaining a healthy lifestyle, getting adequate sleep, positive self-talk, and seeking support from teachers or counselors when needed.", "category": "Mental Health", "keywords": ["exam anxiety", "stress management", "test preparation", "mental health", "relaxation"]},
    {"question": "What is the importance of lifelong learning?", "answer": "Lifelong learning keeps you adaptable in a changing world, enhances career prospects, stimulates mental growth, increases personal satisfaction, helps you stay current with technology and trends, and contributes to better decision-making and problem-solving abilities.", "category": "Personal Development", "keywords": ["lifelong learning", "continuous education", "professional development", "growth", "adaptation"]},
    {"question": "How do I develop better communication skills?", "answer": "Develop communication skills by practicing active listening, reading regularly to expand vocabulary, joining speaking clubs like Toastmasters, seeking feedback, observing effective communicators, practicing empathy, and engaging in diverse conversations with different people.", "category": "Soft Skills", "keywords": ["communication skills", "active listening", "speaking", "presentation", "interpersonal"]}
]

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

class TextPreprocessor:
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
        self.lemmatizer = WordNetLemmatizer()
        self.nlp = nlp
    
    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def tokenize_and_lemmatize(self, text):
        try:
            tokens = word_tokenize(text)
            tokens = [
                self.lemmatizer.lemmatize(token)
                for token in tokens
                if token not in self.stop_words and token not in string.punctuation
            ]
        except:
            tokens = text.split()
        return tokens
    
    def spacy_preprocess(self, text):
        try:
            doc = self.nlp(text)
            tokens = [
                token.lemma_
                for token in doc
                if not token.is_stop and not token.is_punct and not token.is_space
            ]
        except:
            tokens = text.split()
        return tokens
    
    def preprocess_text(self, text, method='spacy'):
        cleaned_text = self.clean_text(text)
        
        if method == 'spacy':
            tokens = self.spacy_preprocess(cleaned_text)
        else:
            tokens = self.tokenize_and_lemmatize(cleaned_text)
        
        processed_text = ' '.join(tokens)
        return processed_text

class FAQMatcher:
    def __init__(self, faqs_data):
        self.faqs = faqs_data
        self.preprocessor = TextPreprocessor()
        
        self.sentence_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                pass
        
        self.processed_questions = []
        self.processed_answers = []
        self.original_faqs = []
        
        for faq in self.faqs:
            processed_q = self.preprocessor.preprocess_text(faq['question'])
            processed_a = self.preprocessor.preprocess_text(faq['answer'])
            
            self.processed_questions.append(processed_q)
            self.processed_answers.append(processed_a)
            self.original_faqs.append(faq)
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.processed_questions)
        
        self.question_embeddings = None
        if self.sentence_model:
            try:
                self.question_embeddings = self.sentence_model.encode(self.processed_questions)
            except Exception as e:
                pass
    
    def tfidf_similarity(self, user_question):
        processed_question = self.preprocessor.preprocess_text(user_question)
        user_tfidf = self.tfidf_vectorizer.transform([processed_question])
        similarities = cosine_similarity(user_tfidf, self.tfidf_matrix).flatten()
        return similarities
    
    def sentence_similarity(self, user_question):
        if not self.sentence_model or self.question_embeddings is None:
            return np.zeros(len(self.original_faqs))
        
        try:
            processed_question = self.preprocessor.preprocess_text(user_question)
            user_embedding = self.sentence_model.encode([processed_question])
            similarities = cosine_similarity(user_embedding, self.question_embeddings).flatten()
            return similarities
        except Exception as e:
            return np.zeros(len(self.original_faqs))
    
    def keyword_matching(self, user_question):
        user_words = set(self.preprocessor.preprocess_text(user_question).split())
        user_question_lower = user_question.lower()
        similarities = []
        
        for faq in self.original_faqs:
            faq_keywords = set()
            
            if 'keywords' in faq:
                for keyword in faq['keywords']:
                    faq_keywords.update(self.preprocessor.preprocess_text(keyword).split())
            
            faq_keywords.update(self.preprocessor.preprocess_text(faq['question']).split())
            
            intersection = len(user_words.intersection(faq_keywords))
            union = len(user_words.union(faq_keywords))
            
            jaccard_similarity = intersection / union if union > 0 else 0
            
            partial_match_score = 0
            for keyword in faq.get('keywords', []):
                if keyword.lower() in user_question_lower:
                    partial_match_score += 0.3
            
            for word in faq['question'].lower().split():
                if len(word) > 3 and word in user_question_lower:
                    partial_match_score += 0.1
            
            final_similarity = min(1.0, jaccard_similarity + partial_match_score)
            similarities.append(final_similarity)
        
        return np.array(similarities)
    
    def find_best_match(self, user_question, threshold=0.1):
        tfidf_sim = self.tfidf_similarity(user_question)
        sentence_sim = self.sentence_similarity(user_question)
        keyword_sim = self.keyword_matching(user_question)
        
        if SENTENCE_TRANSFORMERS_AVAILABLE and self.sentence_model:
            similarities = (0.3 * tfidf_sim + 0.5 * sentence_sim + 0.2 * keyword_sim)
        else:
            similarities = (0.5 * tfidf_sim + 0.5 * keyword_sim)
        
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        if best_score < threshold:
            return None, best_score
        
        return self.original_faqs[best_idx], best_score

class OpenRouterAPI:
    def __init__(self):
        self.api_key = "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"
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.enabled = True  
    
    def enhance_response(self, user_question, faq_answer, context=""):
        if not self.enabled:
            return faq_answer
        
        try:
            prompt = f"""
            You are S.A.M.E.L, an intelligent FAQ assistant created by Hamza Younas (Machine Learning Engineer) for Apexify Technologies. Enhance this response:
            
            Question: {user_question}
            Base Answer: {faq_answer}
            Context: {context}
            
            Provide a clear, concise, and helpful response.
            """
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "microsoft/wizardlm-2-8x22b",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500,
                "temperature": 0.7
            }
            
            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                return faq_answer
                
        except Exception as e:
            return faq_answer
    
    def generate_answer(self, user_question, context=""):
        if not self.enabled:
            return "I apologize, but I couldn't find a relevant answer in my knowledge base. Could you please rephrase your question or ask about a different topic?"
        
        try:
            prompt = f"""
            You are S.A.M.E.L (Smart Automated Machine for Efficient Learning), an intelligent AI assistant specializing in technical and educational topics created by Hamza Younas (Machine Learning Engineer) for Apexify Technologies.
            
            Question: {user_question}
            Context: {context}
            
            Please provide a comprehensive, accurate, and helpful answer to this question. Keep your response:
            - Clear and well-structured
            - Educational and informative
            - Concise but thorough
            - Professional in tone
            
            If the question is technical, provide practical examples or explanations.
            If it's educational, offer actionable advice or learning strategies.
            """
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "microsoft/wizardlm-2-8x22b",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 700,
                "temperature": 0.7
            }
            
            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                return "I apologize, but I'm having trouble processing your request at the moment. Please try again later."
                
        except Exception as e:
            return "I apologize, but I'm having trouble processing your request at the moment. Please try again later."

class SAMELChatbot:
    def __init__(self, domain='technical'):
        self.domain = domain
        
        if domain == 'technical':
            self.faqs_data = TECHNICAL_FAQS
            self.domain_name = "Technical"
        else:
            self.faqs_data = EDUCATIONAL_FAQS
            self.domain_name = "Educational"
        
        self.matcher = FAQMatcher(self.faqs_data)
        self.api = OpenRouterAPI()
        self.chat_history = []
    
    def process_question(self, user_question):
        start_time = time.time()
        
        best_match, confidence = self.matcher.find_best_match(user_question, threshold=0.1)
        
        if best_match is not None and confidence >= 0.2:
            processing_time = time.time() - start_time
            answer = best_match['answer']
            
            response = {
                'answer': answer,
                'confidence': float(confidence),
                'category': best_match.get('category', 'General'),
                'source': f"Knowledge Base: {best_match['question']}",
                'processing_time': processing_time,
                'suggestions': self.get_category_suggestions(best_match['category'])
            }
        else:
            processing_time = time.time() - start_time
            
            if self.api.enabled:
                api_answer = self.api.generate_answer(user_question, f"Domain: {self.domain_name}")
                response = {
                    'answer': api_answer,
                    'confidence': 0.7,  
                    'category': "AI Generated",
                    'source': "OpenRouter API",
                    'processing_time': processing_time,
                    'suggestions': self.get_random_suggestions()
                }
            else:
                response = {
                    'answer': "I apologize, but I couldn't find a relevant answer in my knowledge base. Could you please rephrase your question or ask about a different topic?",
                    'confidence': 0.0,
                    'category': "No Match",
                    'source': "Knowledge base search failed",
                    'processing_time': processing_time,
                    'suggestions': self.get_random_suggestions()
                }
        
        self.chat_history.append({
            'timestamp': datetime.now(),
            'question': user_question,
            'response': response
        })
        
        return response
    
    def get_category_suggestions(self, category):
        category_faqs = [faq for faq in self.faqs_data if faq.get('category') == category]
        if len(category_faqs) > 3:
            import random
            selected = random.sample(category_faqs, 3)
        else:
            selected = category_faqs
        
        return [faq['question'] for faq in selected]
    
    def get_random_suggestions(self):
        import random
        selected = random.sample(self.faqs_data, min(3, len(self.faqs_data)))
        return [faq['question'] for faq in selected]
    
    def get_all_categories(self):
        categories = set()
        for faq in self.faqs_data:
            if 'category' in faq:
                categories.add(faq['category'])
        return sorted(list(categories))

def typing_animation(text, delay=0.03):
    placeholder = st.empty()
    displayed_text = ""
    
    for char in text:
        displayed_text += char
        placeholder.markdown(f'<div class="typing-text">{displayed_text}<span class="cursor">|</span></div>', unsafe_allow_html=True)
        time.sleep(delay)
    
    placeholder.markdown(f'<div class="message-bot">{text}</div>', unsafe_allow_html=True)

def loading_animation():
    loading_html = """
    <div class="loading-container">
        <div class="quantum-processor">
            <div class="quantum-core"></div>
            <div class="data-stream stream-1"></div>
            <div class="data-stream stream-2"></div>
            <div class="data-stream stream-3"></div>
            <div class="neural-pulse"></div>
        </div>
        <div class="loading-text">
            <span class="ai-name">S.A.M.E.L</span> is thinking and analyzing your question....
        </div>
        <div class="progress-indicator">
            <div class="progress-bar"></div>
        </div>
    </div>
    """
    return loading_html

def thinking_animation():
    thinking_html = """
    <div class="thinking-container">
        <div class="brain-activity">
            <div class="synapse synapse-1"></div>
            <div class="synapse synapse-2"></div>
            <div class="synapse synapse-3"></div>
            <div class="synapse synapse-4"></div>
        </div>
        <div class="thinking-text">
             S.A.M.E.L is thinking...
        </div>
    </div>
    """
    return thinking_html

def main():
    st.set_page_config(
        page_title="S.A.M.E.L - FAQ Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Rajdhani', sans-serif;
        background: 
            radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.3) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(120, 219, 255, 0.3) 0%, transparent 50%),
            linear-gradient(135deg, #0f0f23 0%, #1a1a3a 50%, #2d1b69 100%);
        color: #ffffff;
        overflow-x: hidden;
    }
    
    .main-header {
        text-align: center;
        padding: 3rem 0;
        background: linear-gradient(45deg, #00ffff, #ff00ff, #ffff00, #00ffff);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: hologram 4s ease-in-out infinite;
        position: relative;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(255, 255, 255, 0.1) 50%, transparent 70%);
        animation: scan 3s linear infinite;
        pointer-events: none;
    }
    
    @keyframes hologram {
        0%, 100% { background-position: 0% 50%; filter: hue-rotate(0deg); }
        25% { background-position: 100% 50%; filter: hue-rotate(90deg); }
        50% { background-position: 100% 100%; filter: hue-rotate(180deg); }
        75% { background-position: 0% 100%; filter: hue-rotate(270deg); }
    }
    
    @keyframes scan {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .neural-banner {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(0, 255, 255, 0.3);
        box-shadow: 
            0 0 30px rgba(0, 255, 255, 0.2),
            inset 0 0 30px rgba(255, 0, 255, 0.1);
    }
    
    .neural-banner::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #00ffff, #ff00ff, #ffff00, #00ffff);
        background-size: 400% 400%;
        border-radius: 20px;
        z-index: -1;
        animation: borderGlow 3s ease-in-out infinite;
    }
    
    @keyframes borderGlow {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .chat-box {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border-radius: 25px;
        padding: 1.5rem;
        margin: 2rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 
            0 20px 50px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        height: 500px;
        overflow-y: auto;
        overflow-x: hidden;
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }
    
    .chat-box::-webkit-scrollbar {
        width: 8px;
    }
    
    .chat-box::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    .chat-box::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #00ffff, #ff00ff);
        border-radius: 10px;
    }
    
    .chat-box::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #00ffff, #ffff00);
    }
    
    .empty-chat {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
        color: rgba(255, 255, 255, 0.6);
        text-align: center;
    }
    
    .empty-chat h3 {
        margin: 1rem 0;
        color: #00ffff;
        font-size: 1.5rem;
    }
    
    .message-user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 25px 25px 8px 25px;
        margin: 1rem 0 1rem 20%;
        animation: slideInRight 0.5s ease, glow 2s ease-in-out infinite alternate;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .message-bot {
        background: rgba(255, 255, 255, 0.15);
        color: #ffffff;
        padding: 1.5rem 2rem;
        border-radius: 25px 25px 25px 8px;
        margin: 1rem 20% 1rem 0;
        animation: slideInLeft 0.5s ease;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(0, 255, 255, 0.4);
        box-shadow: 0 10px 30px rgba(0, 255, 255, 0.2);
        font-size: 1rem;
        line-height: 1.6;
    }
    
    .message-processing {
        background: rgba(255, 255, 255, 0.1);
        color: #ffffff;
        padding: 1.5rem 2rem;
        border-radius: 25px 25px 25px 8px;
        margin: 1rem 20% 1rem 0;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(0, 255, 255, 0.3);
        box-shadow: 0 10px 30px rgba(0, 255, 255, 0.1);
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 1rem;
        animation: pulse 2s infinite;
        position: relative;
        overflow: hidden;
    }
    
    .message-processing::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 255, 255, 0.2), transparent);
        animation: shimmer 2s linear infinite;
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .processing-text {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.5rem;
        text-align: center;
        font-weight: 500;
    }
    
    .processing-animation {
        display: flex;
        gap: 5px;
    }
    
    .dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #00ffff;
        animation: bounce 1.4s infinite;
    }
    
    .dot1 { animation-delay: 0s; }
    .dot2 { animation-delay: 0.2s; }
    .dot3 { animation-delay: 0.4s; }
    
    @keyframes bounce {
        0%, 60%, 100% {
            transform: translateY(0);
            opacity: 0.3;
        }
        30% {
            transform: translateY(-15px);
            opacity: 1;
        }
    }
    
    .typing-text {
        background: rgba(255, 255, 255, 0.08);
        color: #ffffff;
        padding: 1.5rem 2rem;
        border-radius: 25px 25px 25px 8px;
        margin: 1rem 20% 1rem 0;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(0, 255, 255, 0.2);
        animation: typing 1s ease-in-out infinite;
    }
    
    .cursor {
        animation: blink 1s infinite;
        color: #00ffff;
        font-weight: bold;
    }
    
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
    }
    
    @keyframes typing {
        0%, 100% { border-color: rgba(0, 255, 255, 0.2); }
        50% { border-color: rgba(0, 255, 255, 0.8); }
    }
    
    @keyframes slideInRight {
        from { transform: translateX(100%) rotateY(45deg); opacity: 0; }
        to { transform: translateX(0) rotateY(0deg); opacity: 1; }
    }
    
    @keyframes slideInLeft {
        from { transform: translateX(-100%) rotateY(-45deg); opacity: 0; }
        to { transform: translateX(0) rotateY(0deg); opacity: 1; }
    }
    
    @keyframes glow {
        from { box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3); }
        to { box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6); }
    }
    
    @keyframes pulse {
        0%, 100% { box-shadow: 0 10px 30px rgba(0, 255, 255, 0.1); }
        50% { box-shadow: 0 15px 40px rgba(0, 255, 255, 0.3); }
    }
    
    .loading-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 3rem;
        background: 
            radial-gradient(circle at 50% 50%, rgba(0, 255, 255, 0.1) 0%, transparent 70%),
            linear-gradient(135deg, rgba(255, 255, 255, 0.03) 0%, rgba(0, 255, 255, 0.08) 100%);
        border-radius: 25px;
        margin: 2rem 0;
        backdrop-filter: blur(20px);
        border: 2px solid rgba(0, 255, 255, 0.3);
        box-shadow: 0 20px 50px rgba(0, 255, 255, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .loading-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: conic-gradient(from 0deg, transparent, rgba(0, 255, 255, 0.3), transparent);
        animation: rotate 3s linear infinite;
        z-index: -1;
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .quantum-processor {
        position: relative;
        width: 120px;
        height: 120px;
        margin-bottom: 2rem;
    }
    
    .quantum-core {
        position: absolute;
        top: 50%;
        left: 50%;
        width: 40px;
        height: 40px;
        background: radial-gradient(circle, #00ffff 0%, #0080ff 100%);
        border-radius: 50%;
        transform: translate(-50%, -50%);
        animation: coreExpand 2s ease-in-out infinite;
        box-shadow: 
            0 0 20px #00ffff,
            0 0 40px #00ffff,
            0 0 60px #00ffff;
    }
    
    @keyframes coreExpand {
        0%, 100% { transform: translate(-50%, -50%) scale(1); }
        50% { transform: translate(-50%, -50%) scale(1.3); }
    }
    
    .data-stream {
        position: absolute;
        width: 4px;
        height: 30px;
        background: linear-gradient(to bottom, #00ffff, transparent);
        border-radius: 2px;
        animation: dataFlow 1.5s ease-in-out infinite;
    }
    
    .stream-1 {
        top: 10px;
        left: 50%;
        transform: translateX(-50%);
        animation-delay: 0s;
    }
    
    .stream-2 {
        top: 50%;
        right: 10px;
        transform: translateY(-50%) rotate(90deg);
        animation-delay: 0.5s;
    }
    
    .stream-3 {
        bottom: 10px;
        left: 50%;
        transform: translateX(-50%) rotate(180deg);
        animation-delay: 1s;
    }
    
    @keyframes dataFlow {
        0%, 100% { opacity: 0.3; transform: translateY(0); }
        50% { opacity: 1; transform: translateY(-10px); }
    }
    
    .neural-pulse {
        position: absolute;
        top: 50%;
        left: 50%;
        width: 80px;
        height: 80px;
        border: 2px solid rgba(0, 255, 255, 0.5);
        border-radius: 50%;
        transform: translate(-50%, -50%);
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0% { 
            transform: translate(-50%, -50%) scale(0.8);
            opacity: 1;
        }
        100% { 
            transform: translate(-50%, -50%) scale(1.5);
            opacity: 0;
        }
    }
    
    .loading-text {
        font-size: 1.3rem;
        color: #00ffff;
        animation: textPulse 2s ease-in-out infinite;
        font-weight: 600;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .ai-name {
        color: #ffff00;
        font-weight: 800;
        text-shadow: 0 0 10px #ffff00;
    }
    
    @keyframes textPulse {
        0%, 100% { opacity: 0.7; }
        50% { opacity: 1; }
    }
    
    .progress-indicator {
        width: 200px;
        height: 4px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 2px;
        overflow: hidden;
    }
    
    .progress-bar {
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, #00ffff, #ff00ff, #ffff00, #00ffff);
        background-size: 200% 100%;
        animation: progressFlow 2s linear infinite;
        border-radius: 2px;
    }
    
    @keyframes progressFlow {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    
    .thinking-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 1.5rem;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 0, 255, 0.3);
    }
    
    .brain-activity {
        position: relative;
        width: 80px;
        height: 40px;
        margin-bottom: 1rem;
    }
    
    .synapse {
        position: absolute;
        width: 8px;
        height: 8px;
        background: #ff00ff;
        border-radius: 50%;
        animation: synapseActive 1.5s ease-in-out infinite;
    }
    
    .synapse-1 { top: 0; left: 0; animation-delay: 0s; }
    .synapse-2 { top: 0; right: 0; animation-delay: 0.3s; }
    .synapse-3 { bottom: 0; left: 20px; animation-delay: 0.6s; }
    .synapse-4 { bottom: 0; right: 20px; animation-delay: 0.9s; }
    
    @keyframes synapseActive {
        0%, 100% { 
            transform: scale(1);
            box-shadow: 0 0 5px #ff00ff;
        }
        50% { 
            transform: scale(1.5);
            box-shadow: 0 0 15px #ff00ff, 0 0 25px #ff00ff;
        }
    }
    
    .thinking-text {
        font-size: 1rem;
        color: #ff00ff;
        font-weight: 500;
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.5rem;
    }
    
    .neural-activity-indicator {
        display: flex;
        gap: 3px;
        margin-top: 0.5rem;
    }
    
    .neural-wave {
        width: 4px;
        height: 20px;
        background: linear-gradient(to top, #00ffff, #ff00ff);
        border-radius: 2px;
        animation: neuralWave 1.5s ease-in-out infinite;
    }
    
    .neural-wave:nth-child(1) { animation-delay: 0s; }
    .neural-wave:nth-child(2) { animation-delay: 0.2s; }
    .neural-wave:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes neuralWave {
        0%, 100% { 
            transform: scaleY(0.3);
            opacity: 0.5;
        }
        50% { 
            transform: scaleY(1);
            opacity: 1;
        }
    }
    
    .creator-banner {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.9) 0%, rgba(118, 75, 162, 0.9) 100%);
        color: white;
        padding: 12px 20px;
        border-radius: 30px;
        font-size: 13px;
        font-weight: 600;
        z-index: 1000;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        animation: float 4s ease-in-out infinite, borderPulse 3s ease-in-out infinite;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .creator-banner:hover {
        transform: translateY(-5px) scale(1.05);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6);
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotateZ(0deg); }
        25% { transform: translateY(-10px) rotateZ(1deg); }
        50% { transform: translateY(-5px) rotateZ(0deg); }
        75% { transform: translateY(-15px) rotateZ(-1deg); }
    }
    
    @keyframes borderPulse {
        0%, 100% { border-color: rgba(255, 255, 255, 0.2); }
        50% { border-color: rgba(0, 255, 255, 0.8); }
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 255, 255, 0.1), transparent);
        transition: left 0.5s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-10px) rotateX(5deg);
        box-shadow: 0 20px 50px rgba(102, 126, 234, 0.4);
        border-color: rgba(0, 255, 255, 0.5);
    }
    
    .metric-card:hover::before {
        left: 100%;
    }
    
    .sidebar-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .sidebar-card:hover {
        border-color: rgba(0, 255, 255, 0.3);
        box-shadow: 0 10px 25px rgba(0, 255, 255, 0.1);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.8rem 2.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.5);
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stTextInput > div > div > input {
        background: 
            linear-gradient(135deg, rgba(0, 255, 255, 0.1) 0%, rgba(255, 0, 255, 0.05) 100%);
        color: #ffffff;
        border: 2px solid transparent;
        border-radius: 25px;
        padding: 1.2rem 2rem;
        font-size: 1.1rem;
        font-weight: 500;
        transition: all 0.4s ease;
        backdrop-filter: blur(15px);
        box-shadow: 
            0 8px 30px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        position: relative;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: rgba(0, 255, 255, 0.7);
        font-style: italic;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        background: 
            linear-gradient(135deg, rgba(0, 255, 255, 0.15) 0%, rgba(255, 0, 255, 0.1) 100%);
        border: 2px solid #00ffff;
        box-shadow: 
            0 0 30px rgba(0, 255, 255, 0.4),
            0 0 60px rgba(0, 255, 255, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        outline: none;
        transform: translateY(-2px);
    }
    
    .stTextInput > div > div > input:focus::placeholder {
        color: rgba(0, 255, 255, 0.9);
        transform: translateX(5px);
    }
    
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #00ff88;
        animation: statusPulse 2s infinite;
        margin-right: 10px;
        position: relative;
    }
    
    .status-indicator::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(0, 255, 136, 0.3) 0%, transparent 70%);
        animation: statusGlow 2s infinite;
    }
    
    @keyframes statusPulse {
        0%, 100% { 
            transform: scale(1);
            box-shadow: 0 0 0 0 rgba(0, 255, 136, 0.7);
        }
        50% { 
            transform: scale(1.2);
            box-shadow: 0 0 0 10px rgba(0, 255, 136, 0);
        }
    }
    
    @keyframes statusGlow {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .stDeployButton {display: none;}
    footer {visibility: hidden;}
    .stApp > header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="creator-banner" title="Created by Hamza Younas">
        Created by Hamza Younas
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="main-header">
        <h1 style="font-family: 'Orbitron', monospace; font-size: 4rem; font-weight: 900; margin: 0;">S.A.M.E.L</h1>
        <p style="font-size: 1.4rem; margin: 1rem 0 0 0; opacity: 0.9; font-weight: 500;">Smart Automated Machine for Efficient Learning</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="neural-banner">
        <div style="text-align: center;">
            <h2 style="margin: 0; font-weight: 700; font-size: 2rem; color: #00ffff;">Intelligent FAQ Assistant</h2>
            <p style="margin: 1rem 0 0 0; opacity: 0.9; font-size: 1.1rem;">Get instant answers to technical and educational questions with AI-powered precision</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.markdown("### System Configuration")
        
        domain = st.selectbox(
            "Knowledge Domain",
            options=['technical', 'educational'],
            format_func=lambda x: f"{x.title()} Domain",
            index=0
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.markdown("### System Status")
        st.markdown('<span class="status-indicator"></span>Neural Engine: Active', unsafe_allow_html=True)
        st.markdown('<span class="status-indicator"></span>Language Model: Online', unsafe_allow_html=True)
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            st.markdown('<span class="status-indicator"></span>Embeddings: Advanced', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-indicator"></span>Embeddings: Standard', unsafe_allow_html=True)
        st.markdown('<span class="status-indicator"></span>AI Core: Operational', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.markdown("### Knowledge Base")
        if domain == 'technical':
            st.metric("Total FAQs", len(TECHNICAL_FAQS))
            categories = set(faq['category'] for faq in TECHNICAL_FAQS)
        else:
            st.metric("Total FAQs", len(EDUCATIONAL_FAQS))
            categories = set(faq['category'] for faq in EDUCATIONAL_FAQS)
        
        st.metric("Categories", len(categories))
        st.markdown('</div>', unsafe_allow_html=True)

    if ('chatbot' not in st.session_state or st.session_state.get('domain') != domain):
        st.session_state.chatbot = SAMELChatbot(domain=domain)
        st.session_state.domain = domain
        st.session_state.messages = []
    
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "processing":
        user_question = st.session_state.messages[-2]["content"]
        
        st.markdown("""
        <div class="thinking-container">
            <div class="brain-activity">
                <div class="synapse synapse-1"></div>
                <div class="synapse synapse-2"></div>
                <div class="synapse synapse-3"></div>
                <div class="synapse synapse-4"></div>
            </div>
            <div class="thinking-text">
                <span class="ai-name">S.A.M.E.L</span> is analyzing your question through neural pathways...
                <div class="neural-activity-indicator">
                    <span class="neural-wave"></span>
                    <span class="neural-wave"></span>
                    <span class="neural-wave"></span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        time.sleep(1)
        
        st.session_state.messages.pop()
        response = st.session_state.chatbot.process_question(user_question)
        st.session_state.messages.append({"role": "bot", "content": response['answer']})
        st.session_state.last_response = response
        st.rerun()

    chat_html = '<div class="chat-box">'
    
    if st.session_state.messages:
        for message in st.session_state.messages:
            if message["role"] == "user":
                chat_html += f'<div class="message-user"><strong>You:</strong> {message["content"]}</div>'
            elif message["role"] == "processing":
                chat_html += '''
                <div class="message-processing">
                    <div class="quantum-processor">
                        <div class="quantum-core"></div>
                        <div class="data-stream stream-1"></div>
                        <div class="data-stream stream-2"></div>
                        <div class="data-stream stream-3"></div>
                        <div class="neural-pulse"></div>
                    </div>
                    <div class="processing-text">
                        <span class="ai-name">S.A.M.E.L</span> is processing through quantum neural pathways...
                        <div class="processing-animation">
                            <div class="dot dot1"></div>
                            <div class="dot dot2"></div>
                            <div class="dot dot3"></div>
                        </div>
                    </div>
                </div>
                '''
            else:
                chat_html += f'<div class="message-bot"><strong>S.A.M.E.L:</strong> {message["content"]}</div>'
    else:
        chat_html += '''
        <div class="empty-chat">
            <h3>Welcome to S.A.M.E.L</h3>
            <p>Start a conversation by typing your question below or click on the example questions.</p>
        </div>
        '''
    
    chat_html += '</div>'
    
    st.markdown(chat_html, unsafe_allow_html=True)

    col1, col2 = st.columns([8, 2])
    
    with col1:
        with st.form(key='chat_form', clear_on_submit=True):
            user_question = st.text_input(
                "Question Input", 
                placeholder="Start Chatting with S.A.M.E.L",
                key="user_input",
                label_visibility="collapsed"
            )
            send_button = st.form_submit_button("Send", use_container_width=True)
    
    with col2:
        clear_button = st.button("Reset", use_container_width=True)

    if clear_button:
        st.session_state.messages = []
        st.session_state.chatbot.chat_history = []
        if hasattr(st.session_state, 'last_response'):
            st.session_state.last_response = None
        st.rerun()

    if send_button and user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        st.session_state.messages.append({"role": "processing", "content": "processing"})
        st.rerun()
    
    if hasattr(st.session_state, 'last_response') and st.session_state.last_response:
        with st.expander("🔬 Response Details", expanded=False):
            response = st.session_state.last_response
            
            source_icon = "📚" if "Knowledge Base" in response.get('source', '') else "🤖"
            source_type = "Knowledge Base" if "Knowledge Base" in response.get('source', '') else "AI Generated"
            st.markdown(f"**{source_icon} Answer Source:** {source_type}")
            st.markdown(f"**📍 Details:** {response.get('source', 'Unknown')}")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Confidence", f"{response['confidence']:.1%}")
            with col2:
                st.metric("Category", response['category'])
            with col3:
                st.metric("Response Time", f"{response['processing_time']:.3f}s")
            with col4:
                quality = "Excellent" if response['confidence'] > 0.7 else "Good" if response['confidence'] > 0.4 else "Fair"
                st.metric("Quality", quality)
            
            if response.get('suggestions'):
                st.markdown("**💡 Related Questions:**")
                for i, suggestion in enumerate(response['suggestions']):
                    if st.button(suggestion, key=f"sug_detail_{hash(suggestion)}"):
                        st.session_state.messages.append({"role": "user", "content": suggestion})
                        st.session_state.messages.append({"role": "processing", "content": "processing"})
                        st.rerun()

    st.markdown("### ❓ Frequent Asked Questions")
    if domain == 'technical':
        sample_questions = [
            "What is machine learning?",
            "Who created you?",
            "What is S.A.M.E.L?",
            "What company are you made for?"
        ]
    else:
        sample_questions = [
            "What are effective study techniques?",
            "How can I improve my critical thinking skills?",
            "What is the best way to manage time as a student?",
            "How do I develop better communication skills?"
        ]
    
    cols = st.columns(2)
    for i, question in enumerate(sample_questions):
        with cols[i % 2]:
            if st.button(question, key=f"sample_{i}"):
                st.session_state.messages.append({"role": "user", "content": question})
                st.session_state.messages.append({"role": "processing", "content": "processing"})
                st.rerun()

if __name__ == "__main__":
    main()
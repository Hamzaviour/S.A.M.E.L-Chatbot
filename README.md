# 🤖 S.A.M.E.L - Smart Automated Machine for Efficient Learning

An intelligent FAQ chatbot with a beautiful futuristic Streamlit interface, featuring advanced NLP processing and multiple knowledge domains.

## ✨ Features

- **Dual Knowledge Domains**: Technical and Educational FAQ databases
- **Advanced NLP Processing**: NLTK + spaCy integration for intelligent text understanding
- **Multi-Algorithm Matching**: TF-IDF + Keyword matching ensemble approach
- **Beautiful UI**: Futuristic gradient design with responsive chat interface
- **Real-time Analytics**: Confidence scoring and performance metrics
- **Intelligent Suggestions**: Context-aware related question recommendations
- **Fast Performance**: Optimized for sub-second response times

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd samel-chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download spaCy language model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Run the application**
   ```bash
   streamlit run samel_chatbot_simplified.py
   ```

5. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, manually navigate to the URL shown in the terminal

## 🧪 Testing

Run the test suite to verify functionality:

```bash
python test_samel.py
```

## 📁 Project Structure

```
samel-chatbot/
├── samel_faq_chatbot.ipynb       # Jupyter notebook
├── app.py                 # Test suite
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🔧 Technical Architecture

### Components

1. **TextPreprocessor**: Advanced text cleaning and tokenization
   - NLTK tokenization and lemmatization
   - spaCy named entity recognition
   - Stop word removal and text normalization

2. **FAQMatcher**: Intelligent similarity matching
   - TF-IDF vectorization with cosine similarity
   - Keyword-based Jaccard similarity
   - Ensemble scoring for optimal results

3. **SAMELChatbot**: Main chatbot orchestrator
   - Domain-specific knowledge bases
   - Chat history management
   - Response generation and suggestion system

### Knowledge Domains

#### Technical Domain (8 Categories)
- AI/ML
- Cloud Computing
- Web Security
- Software Development
- Database
- Development Tools
- DevOps
- Security

#### Educational Domain (8 Categories)
- Study Methods
- Cognitive Skills
- Time Management
- Career Guidance
- Online Education
- Mental Health
- Personal Development
- Soft Skills

## 🎯 Performance Metrics

- **Response Time**: < 0.5s average
- **Accuracy**: High confidence matching with threshold controls
- **Scalability**: Efficient vector caching for repeated queries
- **Memory Usage**: Optimized for standard hardware

## 🎨 UI Features

- **Futuristic Design**: Gradient backgrounds and modern styling
- **Responsive Layout**: Adapts to different screen sizes
- **Real-time Chat**: Instant message display with typing indicators
- **Analytics Dashboard**: Confidence scores and category breakdown
- **Interactive Suggestions**: Clickable related questions
- **Domain Switching**: Toggle between Technical and Educational modes

## 🔍 Usage Examples

### Technical Questions
- "What is machine learning?"
- "How does cloud computing work?"
- "Explain the difference between HTTP and HTTPS"
- "What is an API?"

### Educational Questions
- "What are effective study techniques?"
- "How can I improve my critical thinking?"
- "What are the benefits of online learning?"
- "How do I manage time as a student?"

## 🛠️ Customization

### Adding New FAQs

Edit the `TECHNICAL_FAQS` or `EDUCATIONAL_FAQS` lists in `samel_chatbot_simplified.py`:

```python
{
    "question": "Your question here?",
    "answer": "Your detailed answer here.",
    "category": "Category Name",
    "keywords": ["keyword1", "keyword2", "keyword3"]
}
```

### Adjusting Matching Sensitivity

Modify the threshold in the `find_best_match` method:

```python
# Lower threshold = more permissive matching
# Higher threshold = stricter matching
threshold=0.1  # Default value
```

### UI Customization

Modify the CSS in the `st.markdown()` section for custom styling.

## 🐛 Troubleshooting

### Common Issues

1. **spaCy model not found**
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **NLTK data missing**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

3. **Port already in use**
   ```bash
   streamlit run samel_chatbot_simplified.py --server.port 8502
   ```

## 📊 Analytics

The application provides built-in analytics:

- **Confidence Scores**: Measure response reliability
- **Category Distribution**: Track question patterns
- **Response Times**: Monitor performance
- **Chat History**: Review conversation flow

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Test thoroughly
5. Submit a pull request


## 🙏 Acknowledgments

- **NLTK**: Natural Language Toolkit for text processing
- **spaCy**: Industrial-strength NLP library
- **Streamlit**: Beautiful web app framework
- **scikit-learn**: Machine learning utilities
- **NumPy & Pandas**: Data manipulation libraries


**Built by Hamza Younas**


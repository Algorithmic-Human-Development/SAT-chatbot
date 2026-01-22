# SAT Chatbot - LLM-Empowered Therapeutic Dialogue System

A comprehensive therapeutic chatbot system implementing Structured Agentic Therapy (SAT) using Large Language Models (LLMs). This project includes both frontend and backend components for delivering therapeutic dialogue, along with data analysis tools for user study evaluation.

## 📋 Overview

This repository contains the complete implementation of an LLM-based therapeutic chatbot that features:

- **Multi-Agent Architecture**: Structured therapeutic dialogue with specialized agents
- **Single-Agent Mode**: Simplified therapeutic interaction
- **Placebo/Control Mode**: For comparative user studies
- **Data Analysis Tools**: Statistical analysis of user study results

The system is designed to provide evidence-based therapeutic interventions through natural language conversations, incorporating RAG (Retrieval-Augmented Generation) for exercise recommendations and multi-stage dialogue management.

## 🏗️ Repository Structure

```
SAT-chatbot/
├── frontend/              # React-based web application
│   ├── src/
│   │   ├── Chatbot.js            # Multi-agent chatbot implementation
│   │   ├── ChatbotSimple.js      # Single-agent chatbot
│   │   ├── ChatbotPlacebo.js     # Placebo/control chatbot
│   │   ├── Auth/                 # Authentication components
│   │   ├── assets/               # Images and media files
│   │   └── utils/                # Utility functions
│   └── package.json
│
├── backend/               # Django REST API
│   ├── api/
│   │   ├── models.py             # Database models
│   │   ├── views.py              # API endpoints
│   │   ├── serializers.py        # Data serialization
│   │   └── bot/
│   │       ├── gpt.py                    # Main chatbot logic
│   │       ├── simple_bot.py             # Single-agent implementation
│   │       ├── placebo_bot.py            # Placebo implementation
│   │       ├── Memory/                   # Conversation memory management
│   │       ├── Prompts/                  # LLM prompt templates
│   │       └── RAG/                      # Exercise recommendation system
│   ├── backend/           # Django settings
│   ├── manage.py
│   └── requirements.txt
│
└── data-analysis/         # User study analysis scripts
    ├── analyze_sat_user_study.py      # Statistical analysis (ANOVA, permutation tests)
```

## 🚀 Getting Started

### Prerequisites

- **Node.js** (v14 or higher) and npm
- **Python** (3.8 or higher)
- **OpenAI API Key** (for LLM integration)

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Configure environment variables:
   - Create a `.env` file in the frontend directory (or copy from `.env.example`)
   - Add the following:
     ```
     REACT_APP_BASE_URL=http://localhost:8000
     ```
   - Update the URL if your backend is running on a different host or port

4. Start the development server:
```bash
npm start
```

The frontend will be available at `http://localhost:3000` (React's default port).

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
   - Create a `.env` file in the backend directory (or copy from `.env.example`)
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```
   - The `.env.example` file shows all available configuration options

5. Run database migrations:
```bash
python manage.py migrate
```

6. Create a superuser (optional, for admin access):
```bash
python manage.py createsuperuser
```

7. Start the development server:
```bash
python manage.py runserver
```

The backend API will be available at `http://localhost:8000`.

### Data Analysis Setup

**Note:** The data analysis scripts require CSV files with user study data. These files are not included in the repository.

1. Navigate to the data-analysis directory:
```bash
cd data-analysis
```

2. Install required Python packages:
```bash
pip install -r requirements.txt
```

3. Prepare your data files:
   - `SAT User Study (Responses) - Form Responses 1.csv` - Survey responses
   - `email_username_group_flag.csv` - Email to group mapping

4. Run analysis scripts:
```bash
# For user study statistical analysis
python analyze_sat_user_study.py
```

## 📊 Features

### Frontend Features

- **Multiple Chatbot Modes**:
  - Multi-agent structured therapy (Chatbot.js)
  - Single-agent simplified therapy (ChatbotSimple.js)
  - Placebo/control mode (ChatbotPlacebo.js)

- **User Authentication**: Login and registration system
- **Rich UI**: Built with React, Ant Design, and Tailwind CSS
- **Markdown Support**: Display formatted responses from the chatbot
- **Audio Recording**: Support for voice input (AudioRecorder.js)
- **Exercise Visualization**: Interactive exercise recommendations with images

### Backend Features

- **Multi-Stage Dialogue Management**: 
  - Greeting and name collection
  - Emotion identification and validation
  - Event exploration
  - Exercise recommendation
  - Exercise execution and guidance

- **RAG System**: 
  - 26+ therapeutic exercises with detailed descriptions
  - Semantic search for exercise matching
  - Context-aware recommendations

- **Memory Management**: Maintains conversation context across interactions

- **API Endpoints**:
  - User registration and authentication (JWT)
  - Message handling and chat history
  - Group assignment (control/intervention/placebo)

### Data Analysis Features

- **Statistical Analysis**:
  - One-way ANOVA with permutation testing
  - Effect size calculation (eta-squared)
  - Automated visualization (boxplots)
  
- **Metrics**:
  - Group comparison across multiple survey questions
  - Demographic analysis
  - User engagement metrics

## 🔧 Configuration

### Frontend Configuration

Create `frontend/.env` file:
```
REACT_APP_BASE_URL=http://localhost:8000
```

### Backend Configuration

Edit `backend/.env`:
```
OPENAI_API_KEY=your_openai_api_key
SECRET_KEY=your_django_secret_key
DEBUG=True
```

### API Settings

The backend uses Django REST Framework with JWT authentication. Key settings are in `backend/backend/settings.py`:
- CORS settings for frontend communication
- Database configuration (SQLite by default)
- OpenAI integration settings

## 📚 API Documentation

### Key Endpoints

- `POST /api/register/` - User registration
- `POST /api/login/` - User authentication
- `POST /api/message/` - Send message to chatbot (multi-agent mode)
- `POST /api/simple-chat/` - Send message to simple chatbot
- `POST /api/placebo-chat/` - Send message to placebo chatbot
- `GET /api/chat-history/` - Retrieve chat history (requires authentication)
- `GET /api/user-sessions/` - Get all session IDs for authenticated user
- `POST /api/reset-state/` - Reset state machine for authenticated user
- `POST /api/end-session/` - End current session
- `POST /api/send-audio/` - Send audio file for transcription
- `POST /api/process-buffered/` - Process buffered messages for authenticated user

### Authentication

The API uses JWT (JSON Web Tokens) for authentication. Include the token in the Authorization header:
```
Authorization: Bearer <your_jwt_token>
```

## 🧪 Testing

### Frontend Testing
```bash
cd frontend
npm test
```

### Backend Testing
```bash
cd backend
python manage.py test
```

## 📖 Research & Publications

This system was developed as part of research on therapeutic dialogue design using LLMs. Key aspects include:

- Comparison of multi-agent vs. single-agent architectures
- Evaluation of structured vs. unguided conversations
- User study with control, intervention, and placebo groups

For more details, refer to the research paper included in this project.

## 🔐 Security Notes

- Never commit `.env` files or API keys to version control
- Use environment variables for all sensitive configuration
- Implement proper rate limiting in production
- Enable HTTPS for production deployments

## 🤝 Contributing

This is a research project. If you'd like to contribute or adapt this code:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is intended for research and educational purposes. Please cite the associated research paper if you use this code in your work.

## 🙏 Acknowledgments

- OpenAI for GPT API
- Django and React communities
- All user study participants

---

**Note**: This is a research prototype. For production use, additional security hardening, scalability improvements, and comprehensive testing are recommended.

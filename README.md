# 🚀 AI Resume Analyzer

An AI-powered Resume Analyzer that evaluates resumes against job descriptions using ATS-style scoring, keyword matching, and intelligent suggestions. Built with Flask, MongoDB, and a modern responsive UI.

---

## ✨ Features

* 🔐 JWT Authentication (Register/Login)
* 📄 PDF Resume Parsing (pdfplumber)
* 🧠 AI-based Resume Analysis (TF-IDF + NLP)
* 📊 ATS-style Scoring System
* 🎯 Keyword Matching & Missing Skills Detection
* 💡 Smart Contextual Suggestions
* 🌐 Full-stack (Flask + MongoDB + JS frontend)
* 🎨 Modern Dark UI with smooth UX

---

## 🛠 Tech Stack

**Backend:** Flask, MongoDB Atlas, JWT, pdfplumber, scikit-learn
**Frontend:** HTML, CSS, JavaScript
**ML/NLP:** TF-IDF, Cosine Similarity

---

## 📁 Project Structure

```
AI_Resume_Analyzer/
├── backend/
│   ├── app.py
│   ├── analyzer.py
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── css/
│   │   ├── global.css
│   │   ├── auth.css
│   │   └── dashboard.css
│   └── js/
│       └── utils.js
│
└── render.yaml
```

---

## ⚙️ Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/AI_Resume_Analyzer.git
cd AI_Resume_Analyzer/backend
```

---

### 2. Create virtual environment

```bash
py -3.11 -m venv venv
venv\Scripts\activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Configure environment variables

Create `.env` file:

```env
JWT_SECRET_KEY=your_random_secret_key
MONGO_URI=your_mongodb_uri
```

Generate secure key:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

---

### 5. Run backend

```bash
python app.py
```

API runs at:

```
http://127.0.0.1:5000
```

---

### 6. Run frontend

```bash
cd ../frontend
python -m http.server 3000
```

Open:

```
http://localhost:3000
```

---

## 🌐 Deployment

### Backend → Render

* Build Command:

```
pip install -r requirements.txt
```

* Start Command:

```
gunicorn app:app --workers 2 --bind 0.0.0.0:$PORT
```

* Add environment variables in Render dashboard:

  * `MONGO_URI`
  * `JWT_SECRET_KEY`

---

### Frontend → Vercel / Netlify

Update API URL in `frontend/js/utils.js`:

```javascript
const API_URL =
  window.location.hostname === "localhost"
    ? "http://127.0.0.1:5000"
    : "https://your-backend-url.onrender.com";
```

Then deploy frontend folder.

---

## 🔌 API Reference

### POST `/register`

```json
{
  "name": "Rahul",
  "email": "rahul@test.com",
  "password": "123456"
}
```

---

### POST `/login`

```json
{
  "email": "rahul@test.com",
  "password": "123456"
}
```

Response:

```json
{
  "token": "...",
  "message": "Welcome back, Rahul!"
}
```

---

### POST `/analyze` (Protected)

Headers:

```
Authorization: Bearer <TOKEN>
```

Body:

* resume (PDF)
* job_description (text)

---

## 📊 Scoring System

| Factor             | Weight |
| ------------------ | ------ |
| Keyword Similarity | 30%    |
| Skill Match        | 25%    |
| Experience Match   | 20%    |
| Tools Match        | 15%    |
| Resume Quality     | 10%    |

---

## 🎯 Rating Scale

* **85+** → Excellent Match
* **70–84** → Good Match
* **50–69** → Average Match
* **<50** → Low Match

---

## 🔐 Security

* Passwords hashed using bcrypt
* JWT-based authentication
* `.env` not committed (sensitive data protected)

---

## 📌 Future Improvements

* AI resume rewriting (LLM integration)
* Real-time job matching
* Multi-language support
* Resume history tracking

---

## 👨‍💻 Author

Rahul

---

## 📄 License

MIT License

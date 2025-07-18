# 📈 ZeroRiskTrader

**ZeroRiskTrader** is a web-based paper trading platform built with Django, MySQL, and Docker. It allows users to simulate stock trades, track portfolios, and view data-driven insights — all securely deployed on Azure using Nginx and GitHub Actions.

---

## 🔧 Features

- ✅ Paper trading simulator with no real financial risk  
- 📁 Daily stock performance logging and portfolio view  
- 🔐 JWT-based authentication with secure user sessions  
- ⚡ Deployed on Azure with Docker & Nginx (SSL enabled via Let’s Encrypt)

---

## 🧰 Tech Stack

| Layer        | Tools Used                                     |
|--------------|------------------------------------------------|
| Backend      | Python 3.10, Django, Django REST Framework     |
| Database     | MySQL (Azure Managed Instance)                 |
| Frontend     | HTML, CSS, Bootstrap, Django Templates         |
| Deployment   | Docker, Nginx, Azure Web App, Certbot          |
| CI/CD        | GitHub Actions                                 |
| API          | Alpha Vantage API                              |

---

## ⚙️ Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/ankushp1650/zero-risk-trader.git
cd zero-risk-trader

### 2. Set Up Environment Variables

Create a `.env` file in the root directory and configure the following variables:

```env
SECRET_KEY=your_django_secret_key
DEBUG=True
DB_NAME=your_database_name
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_HOST=your_db_host
ALPHA_VANTAGE_API_KEY=your_api_key
```

> 💡 You can also refer to `.env.example` for a sample format.

### 3. Build & Run with Docker

```bash
docker-compose build
docker-compose up
```

App will be live at: [http://localhost:8000](http://localhost:8000)

---

## 🧪 Running Tests

Run unit tests using:

```bash
python manage.py test
```

---

## 📂 Project Structure

```
zero-risk-trader/
│
├── core/                # Django project settings
├── trader/              # Main application logic
├── templates/           # HTML templates for the frontend
├── static/              # CSS and JS files
├── Dockerfile           # Docker image definition
├── docker-compose.yml   # Multi-container setup
├── .env.example         # Sample environment variables
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```


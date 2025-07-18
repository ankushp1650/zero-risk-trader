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

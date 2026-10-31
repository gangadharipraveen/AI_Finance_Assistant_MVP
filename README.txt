AI FINANCE ASSISTANT - QUICK START

1. Extract this ZIP into your AI_Finance_Assistant folder.
2. Open that folder in VS Code (File > Open Folder).
3. Open Terminal > New Terminal.
4. Run: py app.py
5. Open your browser at: http://127.0.0.1:5000
6. Create an account and start adding sample transactions.

If dependencies are missing, run:
py -m pip install -r requirements.txt

Features:
- Register/login with hashed passwords
- Add and delete income/expense transactions
- ML expense categorization using TF-IDF + Naive Bayes
- Monthly dashboard and spending breakdown
- Monthly category budgets and budget warnings
- Basic rule-based financial insights

Demo limitations:
- The ML model uses a small built-in educational dataset. Verify/edit categories.
- This is a local educational MVP, not production financial software.
- Do not enter real sensitive banking credentials or data.
- Insights are educational, not professional financial advice.

from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3, os, re
from datetime import date, datetime
from functools import wraps

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-secret-for-real-deployment")
DB_PATH = os.path.join(os.path.dirname(__file__), "finance.db")

# Small built-in educational training dataset. Improve it with reviewed examples later.
TRAINING = {
    "Food": ["swiggy dinner", "zomato lunch", "pizza burger restaurant", "grocery vegetables fruits", "coffee tea snacks", "food delivery", "restaurant bill", "milk bread eggs"],
    "Transport": ["uber ride", "ola cab", "bus ticket", "metro train fare", "petrol diesel fuel", "auto rickshaw", "parking toll"],
    "Shopping": ["amazon purchase", "clothes shoes shopping", "mall purchase", "online shopping", "electronics purchase"],
    "Bills": ["electricity bill", "water bill", "internet broadband", "mobile recharge", "gas cylinder", "phone bill"],
    "Entertainment": ["netflix subscription", "movie ticket", "concert music", "game subscription", "theatre ticket"],
    "Health": ["hospital medicine", "doctor consultation", "pharmacy tablets", "medical test", "clinic appointment"],
    "Education": ["college fees", "tuition fees", "books stationery", "online course", "exam fee"],
    "Other": ["miscellaneous payment", "cash withdrawal", "unknown expense", "general payment"]
}
_train_x, _train_y = [], []
for category, examples in TRAINING.items():
    for example in examples:
        _train_x.append(example)
        _train_y.append(category)
classifier = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
    ("model", MultinomialNB())
])
classifier.fit(_train_x, _train_y)

def predict_category(description):
    return classifier.predict([description])[0]

def connect_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with connect_db() as db:
        db.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL, email TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL
        )""")
        db.execute("""CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL, description TEXT NOT NULL, amount REAL NOT NULL,
            kind TEXT NOT NULL, category TEXT NOT NULL, trans_date TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )""")
        db.execute("""CREATE TABLE IF NOT EXISTS budgets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL, month TEXT NOT NULL, category TEXT NOT NULL,
            limit_amount REAL NOT NULL,
            UNIQUE(user_id, month, category)
        )""")

def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    return wrapper

@app.route("/")
def home():
    return redirect(url_for("dashboard") if "user_id" in session else url_for("login"))

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not name or not email or len(password) < 6:
            flash("Enter your name and email. Password must be at least 6 characters.", "error")
        else:
            try:
                with connect_db() as db:
                    db.execute("INSERT INTO users(name,email,password_hash) VALUES(?,?,?)",
                               (name, email, generate_password_hash(password)))
                flash("Account created. Please log in.", "success")
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                flash("That email is already registered.", "error")
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        with connect_db() as db:
            user = db.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["name"] = user["name"]
            return redirect(url_for("dashboard"))
        flash("Incorrect email or password.", "error")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/dashboard")
@login_required
def dashboard():
    month = date.today().strftime("%Y-%m")
    with connect_db() as db:
        rows = db.execute("SELECT * FROM transactions WHERE user_id=? ORDER BY trans_date DESC, id DESC",
                          (session["user_id"],)).fetchall()
        budgets = db.execute("SELECT * FROM budgets WHERE user_id=? AND month=?",
                             (session["user_id"], month)).fetchall()
    income = sum(r["amount"] for r in rows if r["kind"] == "Income" and r["trans_date"].startswith(month))
    expenses = sum(r["amount"] for r in rows if r["kind"] == "Expense" and r["trans_date"].startswith(month))
    categories = {}
    for r in rows:
        if r["kind"] == "Expense" and r["trans_date"].startswith(month):
            categories[r["category"]] = categories.get(r["category"], 0) + r["amount"]
    insights = []
    if categories:
        top = max(categories, key=categories.get)
        insights.append(f"Your highest spending category this month is {top} (₹{categories[top]:,.2f}).")
    if expenses == 0:
        insights.append("Add your first expense to start seeing your spending patterns.")
    elif income and expenses > income:
        insights.append("Your recorded expenses are higher than your recorded income this month. Review your budget.")
    else:
        insights.append(f"You have spent ₹{expenses:,.2f} this month. Keep tracking regularly.")
    budget_status = []
    for b in budgets:
        spent = categories.get(b["category"], 0)
        budget_status.append({"category": b["category"], "limit": b["limit_amount"], "spent": spent,
                              "remaining": b["limit_amount"] - spent})
    return render_template("dashboard.html", income=income, expenses=expenses, balance=income-expenses,
                           rows=rows[:8], categories=categories, insights=insights, budgets=budget_status,
                           name=session.get("name", "User"))

@app.route("/transactions", methods=["GET", "POST"])
@login_required
def transactions():
    if request.method == "POST":
        description = request.form.get("description", "").strip()
        kind = request.form.get("kind", "Expense")
        amount_raw = request.form.get("amount", "")
        trans_date = request.form.get("trans_date") or date.today().isoformat()
        try:
            amount = float(amount_raw)
            datetime.strptime(trans_date, "%Y-%m-%d")
            if not description or amount <= 0 or kind not in ("Income", "Expense"):
                raise ValueError()
            category = request.form.get("category", "").strip()
            if kind == "Expense" and not category:
                category = predict_category(description)
            elif kind == "Income":
                category = "Income"
            with connect_db() as db:
                db.execute("""INSERT INTO transactions(user_id,description,amount,kind,category,trans_date)
                              VALUES(?,?,?,?,?,?)""",
                           (session["user_id"], description, amount, kind, category, trans_date))
            flash(f"Transaction saved. Category: {category}", "success")
            return redirect(url_for("transactions"))
        except (ValueError, TypeError):
            flash("Please enter a valid date and an amount greater than zero.", "error")
    with connect_db() as db:
        rows = db.execute("SELECT * FROM transactions WHERE user_id=? ORDER BY trans_date DESC, id DESC",
                          (session["user_id"],)).fetchall()
    categories = list(TRAINING.keys())
    return render_template("transactions.html", rows=rows, categories=categories, today=date.today().isoformat())

@app.route("/transactions/delete/<int:transaction_id>", methods=["POST"])
@login_required
def delete_transaction(transaction_id):
    with connect_db() as db:
        db.execute("DELETE FROM transactions WHERE id=? AND user_id=?", (transaction_id, session["user_id"]))
    flash("Transaction deleted.", "success")
    return redirect(url_for("transactions"))

@app.route("/budgets", methods=["GET", "POST"])
@login_required
def budgets():
    month = date.today().strftime("%Y-%m")
    if request.method == "POST":
        category = request.form.get("category", "")
        try:
            limit_amount = float(request.form.get("limit_amount", ""))
            if category not in TRAINING or limit_amount <= 0:
                raise ValueError()
            with connect_db() as db:
                db.execute("""INSERT INTO budgets(user_id,month,category,limit_amount) VALUES(?,?,?,?)
                              ON CONFLICT(user_id,month,category) DO UPDATE SET limit_amount=excluded.limit_amount""",
                           (session["user_id"], month, category, limit_amount))
            flash("Budget saved for this month.", "success")
            return redirect(url_for("budgets"))
        except ValueError:
            flash("Choose a category and enter a budget greater than zero.", "error")
    with connect_db() as db:
        rows = db.execute("SELECT * FROM budgets WHERE user_id=? AND month=?", (session["user_id"], month)).fetchall()
        expenses = db.execute("""SELECT category, SUM(amount) AS total FROM transactions
            WHERE user_id=? AND kind='Expense' AND substr(trans_date,1,7)=? GROUP BY category""",
            (session["user_id"], month)).fetchall()
    spent = {r["category"]: r["total"] for r in expenses}
    return render_template("budgets.html", rows=rows, spent=spent, categories=list(TRAINING.keys()), month=month)

if __name__ == "__main__":
    init_db()
    app.run(debug=True)

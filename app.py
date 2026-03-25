from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import os
import json
from datetime import date

app = Flask(__name__)
app.secret_key = "soupkitchen_secret_2026_new"
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False

DATA_DIR = "data/raw"
os.makedirs(DATA_DIR, exist_ok=True)

model = None
feature_names = []

# ---- INIT DATA FILES ----
def init_inventory():
    inv_path = os.path.join(DATA_DIR, "inventory.csv")
    if not os.path.exists(inv_path):
        items = [
            ("Canned Corn", 20), ("Canned Tomatoes", 50), ("Pasta", 40), ("Rice", 30),
            ("Canned Beans", 25), ("Bread", 100), ("Potatoes", 50), ("Carrots", 30),
            ("Onions", 40), ("Canned Chicken", 20), ("Powdered Milk", 15), ("Flour", 25),
            ("Sugar", 20), ("Salt", 10), ("Olive Oil", 8), ("Canned Tuna", 35),
            ("Lentils", 45), ("Oats", 28), ("Apples", 60), ("Bananas", 40),
            ("Cheese", 12), ("Eggs", 100), ("Canned Soup Base", 18), ("Spice Mix", 5),
            ("Frozen Vegetables", 30), ("Ground Beef", 15), ("Chicken Breasts", 25),
            ("Fresh Tomatoes", 35), ("Lettuce", 20), ("Cabbage", 15), ("Bell Peppers", 25),
            ("Garlic", 10), ("Pasta Sauce", 40), ("Cereal", 22), ("Peanut Butter", 18),
            ("Jam", 12), ("Canned Sardines", 30), ("Dried Herbs", 8), ("Yeast", 5),
            ("Baking Powder", 6)
        ]
        df = pd.DataFrame(items, columns=["item", "quantity"])
        df.to_csv(inv_path, index=False)

def init_pending_donations():
    pend_path = os.path.join(DATA_DIR, "pending_donations.csv")
    if not os.path.exists(pend_path):
        df = pd.DataFrame(columns=["donation_id", "donor_name", "email", "item", "quantity", "status"])
        df.to_csv(pend_path, index=False)

def init_menus():
    menu_path = os.path.join(DATA_DIR, "menus.csv")
    if not os.path.exists(menu_path):
        menus = [
            ("Monday",    "Hearty Chicken Soup",       "chicken, carrot, onion, potato, celery, garlic, pasta"),
            ("Tuesday",   "Lentil & Tomato Stew",      "lentils, tomato, onion, garlic, spice mix, olive oil"),
            ("Wednesday", "Rice & Beans Bowl",          "rice, canned beans, onion, garlic, bell peppers, spice mix"),
            ("Thursday",  "Vegetable Pasta",            "pasta, pasta sauce, fresh tomatoes, bell peppers, garlic, dried herbs"),
            ("Friday",    "Tuna & Potato Casserole",   "canned tuna, potatoes, onion, garlic, cheese, flour"),
            ("Saturday",  "Oat Porridge & Fruit",      "oats, apples, bananas, sugar, powdered milk"),
            ("Sunday",    "Beef & Cabbage Stew",       "ground beef, cabbage, carrots, onion, potatoes, garlic, spice mix"),
        ]
        df = pd.DataFrame(menus, columns=["day", "meal_name", "ingredients"])
        df.to_csv(menu_path, index=False)

def init_rsvp():
    rsvp_path = os.path.join(DATA_DIR, "rsvp.csv")
    if not os.path.exists(rsvp_path):
        df = pd.DataFrame(columns=["rsvp_date", "name", "email", "guests", "location", "timestamp"])
        df.to_csv(rsvp_path, index=False)

def init_guest_accounts():
    guests_path = os.path.join(DATA_DIR, "guest_accounts.csv")
    if not os.path.exists(guests_path):
        df = pd.DataFrame(columns=["username", "password", "name", "email", "role"])
        # Seed with cook account
        seed = pd.DataFrame([{
            "username": "cook1",
            "password": "cook2025",
            "name": "Chef Maria",
            "email": "cook@soup.org",
            "role": "cook"
        }])
        df = pd.concat([df, seed], ignore_index=True)
        df.to_csv(guests_path, index=False)

def init_model():
    global model, feature_names
    data_path = os.path.join(DATA_DIR, "soup_kitchen_dataset.csv")
    if not os.path.exists(data_path):
        np.random.seed(42)
        n = 500
        months = np.random.randint(1, 13, n)
        temps = np.random.uniform(20, 100, n)
        rain = np.random.randint(0, 2, n)
        snow = np.random.randint(0, 2, n)
        holiday = np.random.randint(0, 2, n)
        event = np.random.randint(0, 2, n)
        eom = np.random.randint(0, 2, n)
        people = (
            80 +
            (12 - months) * 2 +
            (70 - temps) * 0.5 +
            rain * 15 +
            snow * 25 +
            holiday * 30 +
            event * 20 +
            eom * 18 +
            np.random.normal(0, 10, n)
        ).astype(int)
        people = np.clip(people, 30, 300)
        df = pd.DataFrame({
            "date": pd.date_range("2021-01-01", periods=n, freq="D").astype(str),
            "day_of_week": ["Mon"] * n,
            "month": months, "temp": temps, "rain": rain, "snow": snow,
            "holiday": holiday, "local_event": event, "end_of_month": eom,
            "people": people
        })
        df.to_csv(data_path, index=False)

    data = pd.read_csv(data_path)
    drop_cols = [c for c in ["people", "date", "day_of_week"] if c in data.columns]
    X = data.drop(drop_cols, axis=1)
    y = data["people"]
    feature_names = X.columns.tolist()
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

# Initialize everything
init_inventory()
init_pending_donations()
init_menus()
init_rsvp()
init_guest_accounts()
init_model()

# ---- HELPERS ----
def load_inventory():
    return pd.read_csv(os.path.join(DATA_DIR, "inventory.csv"))

def save_inventory(df):
    df.to_csv(os.path.join(DATA_DIR, "inventory.csv"), index=False)

def load_pending():
    return pd.read_csv(os.path.join(DATA_DIR, "pending_donations.csv"))

def save_pending(df):
    df.to_csv(os.path.join(DATA_DIR, "pending_donations.csv"), index=False)

def load_rsvp():
    return pd.read_csv(os.path.join(DATA_DIR, "rsvp.csv"))

def save_rsvp(df):
    df.to_csv(os.path.join(DATA_DIR, "rsvp.csv"), index=False)

def load_guest_accounts():
    return pd.read_csv(os.path.join(DATA_DIR, "guest_accounts.csv"))

def save_guest_accounts(df):
    df.to_csv(os.path.join(DATA_DIR, "guest_accounts.csv"), index=False)

def get_current_role():
    if session.get('host_logged_in'):
        return 'host'
    if session.get('cook_logged_in'):
        return 'cook'
    if session.get('guest_logged_in'):
        return 'guest'
    return None

# ---- ROUTES ----
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    role = get_current_role()
    if role not in ('host', 'cook'):
        return jsonify({"error": "Unauthorized"}), 401
    data = request.json
    features = np.array([[
        int(data.get('month', 1)),
        float(data.get('temp', 70)),
        int(data.get('rain', 0)),
        int(data.get('snow', 0)),
        int(data.get('holiday', 0)),
        int(data.get('event', 0)),
        int(data.get('eom', 0))
    ]])
    prediction = int(model.predict(features)[0])
    return jsonify({
        "guests": prediction,
        "soup": prediction + 5,
        "bread": prediction + 10
    })

@app.route('/get_menu', methods=['GET'])
def get_menu():
    day = request.args.get('day', 'Monday').capitalize()
    df = pd.read_csv(os.path.join(DATA_DIR, "menus.csv"))
    row = df[df['day'] == day]
    if not row.empty:
        return jsonify({
            "meal": row.iloc[0]['meal_name'],
            "ingredients": row.iloc[0]['ingredients'].split(", ")
        })
    return jsonify({"error": "Not found"}), 404

@app.route('/get_inventory', methods=['GET'])
def get_inventory():
    inv = load_inventory()
    return jsonify(inv.to_dict(orient='records'))

@app.route('/submit_donation', methods=['POST'])
def submit_donation():
    data = request.json
    item = data.get('item', '').strip()
    qty = int(data.get('quantity', 1))
    name = data.get('donor_name', '').strip()
    email = data.get('email', '').strip()
    if not item or not name or not email:
        return jsonify({"error": "All fields required"}), 400
    pend = load_pending()
    new_id = int(pend["donation_id"].max() + 1) if not pend.empty and not pend["donation_id"].isna().all() else 1
    new_row = pd.DataFrame([{
        "donation_id": new_id,
        "donor_name": name,
        "email": email,
        "item": item,
        "quantity": qty,
        "status": "pending"
    }])
    pend = pd.concat([pend, new_row], ignore_index=True)
    save_pending(pend)
    return jsonify({"success": True, "donation_id": new_id})

@app.route('/get_pending', methods=['GET'])
def get_pending():
    pend = load_pending()
    return jsonify(pend.to_dict(orient='records'))

@app.route('/approve_donation', methods=['POST'])
def approve_donation():
    if not session.get('host_logged_in'):
        return jsonify({"error": "Unauthorized"}), 401
    data = request.json
    did = int(data.get('donation_id'))
    pend = load_pending()
    row = pend[pend['donation_id'] == did]
    if row.empty:
        return jsonify({"error": "Not found"}), 404
    r = row.iloc[0]
    inv = load_inventory()
    if r['item'] in inv['item'].values:
        inv.loc[inv['item'] == r['item'], 'quantity'] += int(r['quantity'])
    else:
        inv = pd.concat([inv, pd.DataFrame([{"item": r['item'], "quantity": int(r['quantity'])}])], ignore_index=True)
    save_inventory(inv)
    pend.loc[pend['donation_id'] == did, 'status'] = 'approved'
    save_pending(pend)
    return jsonify({"success": True})

# ---- GUEST ACCOUNTS ----
@app.route('/guest_signup', methods=['POST'])
def guest_signup():
    data = request.json
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    name = data.get('name', '').strip()
    email = data.get('email', '').strip()
    if not username or not password or not name or not email:
        return jsonify({"error": "All fields required"}), 400
    accounts = load_guest_accounts()
    if username in accounts['username'].values:
        return jsonify({"error": "Username already taken"}), 400
    new_row = pd.DataFrame([{
        "username": username, "password": password,
        "name": name, "email": email, "role": "guest"
    }])
    accounts = pd.concat([accounts, new_row], ignore_index=True)
    save_guest_accounts(accounts)
    session['guest_logged_in'] = True
    session['guest_username'] = username
    session['guest_name'] = name
    session['guest_role'] = 'guest'
    return jsonify({"success": True, "name": name, "role": "guest"})

@app.route('/guest_login', methods=['POST'])
def guest_login():
    data = request.json
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    accounts = load_guest_accounts()
    row = accounts[(accounts['username'] == username) & (accounts['password'] == password)]
    if row.empty:
        return jsonify({"error": "Invalid credentials"}), 401
    r = row.iloc[0]
    session['guest_logged_in'] = True
    session['guest_username'] = username
    session['guest_name'] = r['name']
    session['guest_role'] = r['role']
    return jsonify({"success": True, "name": r['name'], "role": r['role']})

@app.route('/guest_logout', methods=['POST'])
def guest_logout():
    session.pop('guest_logged_in', None)
    session.pop('guest_username', None)
    session.pop('guest_name', None)
    session.pop('guest_role', None)
    return jsonify({"success": True})

@app.route('/guest_status', methods=['GET'])
def guest_status():
    return jsonify({
        "logged_in": session.get('guest_logged_in', False),
        "name": session.get('guest_name', ''),
        "role": session.get('guest_role', ''),
        "username": session.get('guest_username', '')
    })

# ---- RSVP ----
@app.route('/rsvp', methods=['POST'])
def submit_rsvp():
    if not session.get('guest_logged_in') and not session.get('host_logged_in') and not session.get('cook_logged_in'):
        return jsonify({"error": "Please log in to RSVP"}), 401
    data = request.json
    rsvp_date = data.get('date', str(date.today()))
    guests_count = int(data.get('guests', 1))
    location = data.get('location', '').strip()
    name = session.get('guest_name', data.get('name', ''))
    email = data.get('email', '')

    rsvp = load_rsvp()
    # Check if already RSVP'd for this date from this user
    username = session.get('guest_username', '')
    existing = rsvp[(rsvp['email'] == email) & (rsvp['rsvp_date'] == rsvp_date)] if email else pd.DataFrame()
    if not existing.empty:
        return jsonify({"error": "You have already RSVP'd for this date"}), 400

    new_row = pd.DataFrame([{
        "rsvp_date": rsvp_date,
        "name": name,
        "email": email,
        "guests": guests_count,
        "location": location,
        "timestamp": pd.Timestamp.now().isoformat()
    }])
    rsvp = pd.concat([rsvp, new_row], ignore_index=True)
    save_rsvp(rsvp)
    return jsonify({"success": True})

@app.route('/get_rsvp', methods=['GET'])
def get_rsvp():
    role = get_current_role()
    if role not in ('host', 'cook'):
        return jsonify({"error": "Unauthorized"}), 401
    rsvp_date = request.args.get('date', str(date.today()))
    rsvp = load_rsvp()
    day_rsvp = rsvp[rsvp['rsvp_date'] == rsvp_date] if not rsvp.empty else pd.DataFrame()
    total_people = int(day_rsvp['guests'].sum()) if not day_rsvp.empty else 0
    return jsonify({
        "date": rsvp_date,
        "total_people": total_people,
        "rsvp_count": len(day_rsvp),
        "entries": day_rsvp.to_dict(orient='records') if not day_rsvp.empty else []
    })

@app.route('/get_rsvp_summary', methods=['GET'])
def get_rsvp_summary():
    role = get_current_role()
    if role not in ('host', 'cook'):
        return jsonify({"error": "Unauthorized"}), 401
    rsvp = load_rsvp()
    if rsvp.empty:
        return jsonify([])
    summary = rsvp.groupby('rsvp_date').agg(
        rsvp_count=('name', 'count'),
        total_guests=('guests', 'sum')
    ).reset_index()
    return jsonify(summary.to_dict(orient='records'))

# ---- HOST & COOK LOGIN ----
@app.route('/host_login', methods=['POST'])
def host_login():
    data = request.json
    if data.get('username') == 'admin' and data.get('password') == 'soupkitchen2025':
        session['host_logged_in'] = True
        return jsonify({"success": True, "role": "host"})
    return jsonify({"error": "Invalid credentials"}), 401

@app.route('/host_logout', methods=['POST'])
def host_logout():
    session.pop('host_logged_in', None)
    return jsonify({"success": True})

@app.route('/host_status', methods=['GET'])
def host_status():
    return jsonify({"logged_in": session.get('host_logged_in', False)})

@app.route('/update_inventory', methods=['POST'])
def update_inventory():
    if not session.get('host_logged_in'):
        return jsonify({"error": "Unauthorized"}), 401
    data = request.json
    item = data.get('item', '').strip()
    qty = int(data.get('quantity', 0))
    inv = load_inventory()
    if item in inv['item'].values:
        inv.loc[inv['item'] == item, 'quantity'] = qty
    else:
        inv = pd.concat([inv, pd.DataFrame([{"item": item, "quantity": qty}])], ignore_index=True)
    save_inventory(inv)
    return jsonify({"success": True})

@app.route('/cook_login', methods=['POST'])
def cook_login():
    data = request.json
    accounts = load_guest_accounts()
    row = accounts[(accounts['username'] == data.get('username')) &
                   (accounts['password'] == data.get('password')) &
                   (accounts['role'] == 'cook')]
    if not row.empty:
        session['cook_logged_in'] = True
        session['cook_name'] = row.iloc[0]['name']
        return jsonify({"success": True, "name": row.iloc[0]['name']})
    return jsonify({"error": "Invalid credentials"}), 401

@app.route('/cook_logout', methods=['POST'])
def cook_logout():
    session.pop('cook_logged_in', None)
    session.pop('cook_name', None)
    return jsonify({"success": True})

@app.route('/cook_status', methods=['GET'])
def cook_status():
    return jsonify({
        "logged_in": session.get('cook_logged_in', False),
        "name": session.get('cook_name', '')
    })

@app.route('/session_info', methods=['GET'])
def session_info():
    """Returns the current user's role and info for the frontend."""
    if session.get('host_logged_in'):
        return jsonify({"role": "host", "name": "Host Admin"})
    if session.get('cook_logged_in'):
        return jsonify({"role": "cook", "name": session.get('cook_name', 'Cook')})
    if session.get('guest_logged_in'):
        return jsonify({"role": session.get('guest_role', 'guest'), "name": session.get('guest_name', '')})
    return jsonify({"role": "visitor", "name": ""})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import os
import json
from datetime import date, timedelta
import random

app = Flask(__name__)
app.secret_key = "soupkitchen_secret_2026_new"
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False

DATA_DIR = "data/raw"
os.makedirs(DATA_DIR, exist_ok=True)

model = None
feature_names = []

# ---- EXPIRY RULES: days from today each category typically lasts ----
EXPIRY_RULES = {
    "Canned Corn": 730, "Canned Tomatoes": 730, "Canned Beans": 730,
    "Canned Chicken": 730, "Canned Tuna": 730, "Canned Soup Base": 730,
    "Canned Sardines": 730, "Pasta Sauce": 365,
    "Pasta": 730, "Rice": 730, "Flour": 365, "Sugar": 730, "Salt": 1825,
    "Oats": 365, "Cereal": 180, "Lentils": 730, "Dried Herbs": 365,
    "Baking Powder": 365, "Yeast": 180, "Spice Mix": 365,
    "Olive Oil": 365, "Peanut Butter": 365, "Jam": 365, "Powdered Milk": 365,
    "Bread": 5, "Apples": 14, "Bananas": 7, "Fresh Tomatoes": 7,
    "Lettuce": 5, "Carrots": 21, "Potatoes": 30, "Onions": 30,
    "Cabbage": 14, "Bell Peppers": 10, "Garlic": 30,
    "Eggs": 21, "Cheese": 21, "Butter": 30,
    "Ground Beef": 3, "Chicken Breasts": 3,
    "Frozen Vegetables": 180,
}

# ---- HOME RECIPE IDEAS per ingredient ----
HOME_RECIPES = {
    "Pasta": [
        {"name": "Aglio e Olio", "desc": "Toss with garlic, olive oil, and parsley for a quick 10-minute dinner."},
        {"name": "Cold Pasta Salad", "desc": "Mix with chopped veggies and a squeeze of lemon — great for lunch."},
        {"name": "Mac & Cheese", "desc": "Stir in cheese and a splash of milk for a comforting bowl."},
    ],
    "Rice": [
        {"name": "Egg Fried Rice", "desc": "Fry leftover rice with an egg, soy sauce, and any vegetables."},
        {"name": "Rice Porridge (Congee)", "desc": "Simmer with extra water until creamy, top with a soft-boiled egg."},
        {"name": "Stuffed Peppers", "desc": "Fill bell peppers with seasoned rice and bake for 20 min."},
    ],
    "Lentils": [
        {"name": "Lentil Soup", "desc": "Simmer with onion, garlic, and cumin for a warming bowl."},
        {"name": "Lentil Patties", "desc": "Mash cooked lentils with spices and pan-fry until golden."},
        {"name": "Lentil Salad", "desc": "Toss cooled lentils with vinegar, mustard, and fresh herbs."},
    ],
    "Oats": [
        {"name": "Overnight Oats", "desc": "Soak in milk overnight, top with banana and honey in the morning."},
        {"name": "Oat Banana Pancakes", "desc": "Blend oats with a ripe banana and egg — fry like pancakes."},
        {"name": "Granola Bars", "desc": "Mix with peanut butter and honey, press into a pan and refrigerate."},
    ],
    "Canned Beans": [
        {"name": "Bean Tacos", "desc": "Season with cumin and chilli, stuff into tortillas with cheese."},
        {"name": "White Bean Soup", "desc": "Blend half the beans with broth for a creamy, hearty soup."},
        {"name": "Bean Salad", "desc": "Drain, rinse, toss with red onion, lemon juice, and olive oil."},
    ],
    "Canned Tuna": [
        {"name": "Tuna Sandwich", "desc": "Mix with a little mayo, lemon, and black pepper on toast."},
        {"name": "Tuna Pasta", "desc": "Toss with pasta, olive oil, capers, and cherry tomatoes."},
        {"name": "Tuna Patties", "desc": "Mix with mashed potato and egg, pan-fry until crispy."},
    ],
    "Canned Tomatoes": [
        {"name": "Simple Tomato Sauce", "desc": "Simmer with garlic and dried herbs for pasta or pizza."},
        {"name": "Shakshuka", "desc": "Poach eggs directly in spiced tomato sauce — serve with bread."},
        {"name": "Tomato Soup", "desc": "Blend with a little cream and basil for a warming bowl."},
    ],
    "Potatoes": [
        {"name": "Roasted Potatoes", "desc": "Cube, toss with olive oil and salt, roast at 200C for 30 min."},
        {"name": "Potato Soup", "desc": "Simmer diced potato with onion and broth until soft, then blend."},
        {"name": "Hash Browns", "desc": "Grate, squeeze dry, fry in a hot pan until golden on both sides."},
    ],
    "Carrots": [
        {"name": "Glazed Carrots", "desc": "Cook in butter and a pinch of sugar until shiny and tender."},
        {"name": "Carrot Soup", "desc": "Roast with onion and ginger, blend with broth for a silky soup."},
        {"name": "Carrot Sticks & Hummus", "desc": "Cut into sticks and dip in any bean-based dip."},
    ],
    "Eggs": [
        {"name": "Soft-Boiled Eggs", "desc": "Boil for exactly 6 minutes, peel carefully — perfect on toast."},
        {"name": "Frittata", "desc": "Whisk with leftover veggies, cook in a pan then bake 10 min."},
        {"name": "Egg Drop Soup", "desc": "Drizzle beaten egg into simmering broth while stirring."},
    ],
    "Bread": [
        {"name": "French Toast", "desc": "Dip slices in egg and milk mixture, fry until golden."},
        {"name": "Bread Crumbs", "desc": "Toast stale bread in oven, blend for a topping or coating."},
        {"name": "Garlic Bread", "desc": "Spread with butter and garlic, bake wrapped in foil for 15 min."},
    ],
    "Cabbage": [
        {"name": "Simple Coleslaw", "desc": "Shred and mix with a little mayo, vinegar, and sugar."},
        {"name": "Stir-Fried Cabbage", "desc": "Fry with garlic and soy sauce over high heat for 5 min."},
        {"name": "Cabbage Rolls", "desc": "Wrap seasoned rice and meat in leaves, bake in tomato sauce."},
    ],
    "Onions": [
        {"name": "Caramelised Onions", "desc": "Cook low and slow in butter for 30 min until golden and sweet."},
        {"name": "French Onion Soup", "desc": "Simmer caramelised onions in broth, top with cheese toast."},
        {"name": "Pickled Onions", "desc": "Soak in vinegar, sugar, and salt for 30 min — great on anything."},
    ],
    "Flour": [
        {"name": "Simple Flatbreads", "desc": "Mix flour, water, and salt into dough, roll thin and fry."},
        {"name": "Pancakes", "desc": "Mix with egg and milk for a quick breakfast batter."},
        {"name": "Thickener for Soups", "desc": "Whisk a spoonful into soups or sauces to make them creamy."},
    ],
    "Apples": [
        {"name": "Baked Apples", "desc": "Core, fill with oats and cinnamon, bake at 180C for 25 min."},
        {"name": "Apple Sauce", "desc": "Simmer chopped apples with a little sugar and lemon until soft."},
        {"name": "Apple & Cheese Snack", "desc": "Slice thinly and pair with any cheese — a classic combo."},
    ],
    "Bananas": [
        {"name": "Banana Nice Cream", "desc": "Freeze ripe bananas, blend until smooth — like ice cream!"},
        {"name": "Banana Bread", "desc": "Mash overripe bananas with flour, egg, and sugar, bake 50 min."},
        {"name": "Banana Porridge", "desc": "Mash half a banana into oatmeal while cooking for natural sweetness."},
    ],
    "Canned Corn": [
        {"name": "Corn Fritters", "desc": "Mix with egg and flour, pan-fry until golden on both sides."},
        {"name": "Corn Chowder", "desc": "Simmer with potato and milk for a creamy, hearty soup."},
        {"name": "Corn Salsa", "desc": "Mix with diced tomato, onion, and lime juice as a fresh topping."},
    ],
    "Peanut Butter": [
        {"name": "Peanut Noodles", "desc": "Mix with soy sauce and a little water, toss with noodles."},
        {"name": "Peanut Butter Cookies", "desc": "Mix with sugar and egg, roll into balls and bake 10 min."},
        {"name": "Smoothie Boost", "desc": "Add a spoonful to any smoothie for protein and creaminess."},
    ],
    "Cheese": [
        {"name": "Cheese Omelette", "desc": "Fold grated cheese into a beaten egg omelette."},
        {"name": "Cheese Toastie", "desc": "Layer between bread and fry in butter until golden and melted."},
        {"name": "Cheese Sauce", "desc": "Melt into a flour and milk roux for mac and cheese or veggies."},
    ],
    "Frozen Vegetables": [
        {"name": "Veggie Stir-Fry", "desc": "Toss straight from frozen into a hot wok with soy sauce."},
        {"name": "Vegetable Soup", "desc": "Add to simmering broth with any herbs — ready in 10 min."},
        {"name": "Veggie Fried Rice", "desc": "Mix with leftover rice, egg, and a splash of soy sauce."},
    ],
}

DEFAULT_RECIPES = [
    {"name": "Simple Stew", "desc": "Chop and simmer with water, salt and any other vegetables you have."},
    {"name": "Stir Fry", "desc": "Cut into pieces and fry in a hot pan with a little oil and seasoning."},
    {"name": "Soup", "desc": "Add to boiling water with salt and herbs for a warming bowl."},
]

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

def init_expiry():
    inv_path = os.path.join(DATA_DIR, "inventory.csv")
    df = pd.read_csv(inv_path)
    if 'expiry_date' not in df.columns:
        today = date.today()
        def get_expiry(item):
            days = EXPIRY_RULES.get(item, 365)
            variation = random.randint(-int(days * 0.15), int(days * 0.15))
            return str(today + timedelta(days=max(1, days + variation)))
        df['expiry_date'] = df['item'].apply(get_expiry)
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
        seed = pd.DataFrame([{
            "username": "cook1", "password": "cook2025",
            "name": "Chef", "email": "cook@soup.org", "role": "cook"
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
            80 + (12 - months) * 2 + (70 - temps) * 0.5 +
            rain * 15 + snow * 25 + holiday * 30 + event * 20 + eom * 18 +
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

random.seed(42)
init_inventory()
init_expiry()
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

def days_until_expiry(expiry_str):
    try:
        exp = date.fromisoformat(str(expiry_str))
        return (exp - date.today()).days
    except:
        return 999

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
        int(data.get('month', 1)), float(data.get('temp', 70)),
        int(data.get('rain', 0)), int(data.get('snow', 0)),
        int(data.get('holiday', 0)), int(data.get('event', 0)),
        int(data.get('eom', 0))
    ]])
    prediction = int(model.predict(features)[0])
    return jsonify({"guests": prediction, "soup": prediction + 5, "bread": prediction + 10})

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
    if 'expiry_date' not in inv.columns:
        inv['expiry_date'] = ''
    return jsonify(inv.to_dict(orient='records'))

@app.route('/get_inventory_with_expiry', methods=['GET'])
def get_inventory_with_expiry():
    role = get_current_role()
    if role not in ('host', 'cook'):
        return jsonify({"error": "Unauthorized"}), 401
    inv = load_inventory()
    if 'expiry_date' not in inv.columns:
        inv['expiry_date'] = ''
    records = []
    for _, row in inv.iterrows():
        days_left = days_until_expiry(row.get('expiry_date', ''))
        if days_left <= 3:
            urgency = "critical"
        elif days_left <= 7:
            urgency = "urgent"
        elif days_left <= 14:
            urgency = "soon"
        else:
            urgency = "ok"
        records.append({
            "item": row['item'],
            "quantity": int(row['quantity']),
            "expiry_date": str(row.get('expiry_date', '')),
            "days_left": days_left,
            "urgency": urgency
        })
    urgency_order = {"critical": 0, "urgent": 1, "soon": 2, "ok": 3}
    records.sort(key=lambda x: (urgency_order[x['urgency']], x['days_left']))
    return jsonify(records)

@app.route('/get_cook_recommendations', methods=['GET'])
def get_cook_recommendations():
    role = get_current_role()
    if role not in ('host', 'cook'):
        return jsonify({"error": "Unauthorized"}), 401
    inv = load_inventory()
    if 'expiry_date' not in inv.columns:
        return jsonify([])
    recommendations = []
    for _, row in inv.iterrows():
        days_left = days_until_expiry(row.get('expiry_date', ''))
        qty = int(row['quantity'])
        item = row['item']
        if days_left <= 3 and qty > 0:
            recommendations.append({
                "item": item, "days_left": days_left, "quantity": qty,
                "priority": "USE TODAY",
                "priority_level": "critical",
                "message": f"Only {days_left} day(s) left! Use {item} in today's meal immediately."
            })
        elif days_left <= 7 and qty > 0:
            recommendations.append({
                "item": item, "days_left": days_left, "quantity": qty,
                "priority": "USE THIS WEEK",
                "priority_level": "urgent",
                "message": f"{item} expires in {days_left} days. Plan a meal using this ingredient soon."
            })
        elif days_left <= 14 and qty > 20:
            recommendations.append({
                "item": item, "days_left": days_left, "quantity": qty,
                "priority": "USE IN LARGE PORTIONS",
                "priority_level": "soon",
                "message": f"You have {qty} units of {item} expiring in {days_left} days. Use in larger portions."
            })
    recommendations.sort(key=lambda x: x['days_left'])
    return jsonify(recommendations[:15])

@app.route('/get_recipes', methods=['GET'])
def get_recipes():
    item = request.args.get('item', '').strip()
    recipes = HOME_RECIPES.get(item, DEFAULT_RECIPES)
    return jsonify({"item": item, "recipes": recipes})

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
        "donation_id": new_id, "donor_name": name,
        "email": email, "item": item, "quantity": qty, "status": "pending"
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
        days = EXPIRY_RULES.get(r['item'], 365)
        new_expiry = str(date.today() + timedelta(days=days))
        new_row = pd.DataFrame([{"item": r['item'], "quantity": int(r['quantity']), "expiry_date": new_expiry}])
        inv = pd.concat([inv, new_row], ignore_index=True)
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
    new_row = pd.DataFrame([{"username": username, "password": password, "name": name, "email": email, "role": "guest"}])
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
    existing = rsvp[(rsvp['email'] == email) & (rsvp['rsvp_date'] == rsvp_date)] if email else pd.DataFrame()
    if not existing.empty:
        return jsonify({"error": "You have already RSVP'd for this date"}), 400
    new_row = pd.DataFrame([{
        "rsvp_date": rsvp_date, "name": name, "email": email,
        "guests": guests_count, "location": location,
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
        "date": rsvp_date, "total_people": total_people,
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
        rsvp_count=('name', 'count'), total_guests=('guests', 'sum')
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
        days = EXPIRY_RULES.get(item, 365)
        new_expiry = str(date.today() + timedelta(days=days))
        inv = pd.concat([inv, pd.DataFrame([{"item": item, "quantity": qty, "expiry_date": new_expiry}])], ignore_index=True)
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
    return jsonify({"logged_in": session.get('cook_logged_in', False), "name": session.get('cook_name', '')})

@app.route('/session_info', methods=['GET'])
def session_info():
    if session.get('host_logged_in'):
        return jsonify({"role": "host", "name": "Host Admin"})
    if session.get('cook_logged_in'):
        return jsonify({"role": "cook", "name": session.get('cook_name', 'Cook')})
    if session.get('guest_logged_in'):
        return jsonify({"role": session.get('guest_role', 'guest'), "name": session.get('guest_name', '')})
    return jsonify({"role": "visitor", "name": ""})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
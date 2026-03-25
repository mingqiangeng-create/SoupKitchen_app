import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Soup Kitchen Assistant", page_icon="🍲", layout="wide")

# --- DATA DIRECTORY & INITIALIZATION ---
DATA_DIR = "data/raw"
os.makedirs(DATA_DIR, exist_ok=True)

@st.cache_resource
def train_model():
    data_path = os.path.join(DATA_DIR, "soup_kitchen_dataset.csv")
    if not os.path.exists(data_path):
        st.error("Missing soup_kitchen_dataset.csv - please add it to data/raw/")
        st.stop()
    data = pd.read_csv(data_path)
    X = data.drop(["people", "date", "day_of_week"], axis=1)
    y = data["people"]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X.columns.tolist()

@st.cache_data
def load_menu():
    menu_path = os.path.join(DATA_DIR, "menus.csv")
    if not os.path.exists(menu_path):
        st.error("Missing menus.csv - please add it to data/raw/")
        st.stop()
    return pd.read_csv(menu_path)

# Initialize inventory and pending donations if they don't exist
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

# Initialize files
init_inventory()
init_pending_donations()

# Load model and menu
try:
    model, feature_names = train_model()
    menu_df = load_menu()
except Exception as e:
    st.error(f"Error loading core datasets: {e}")
    st.stop()

# Load inventory and pending (fresh every rerun)
def load_inventory():
    return pd.read_csv(os.path.join(DATA_DIR, "inventory.csv"))

def save_inventory(df):
    df.to_csv(os.path.join(DATA_DIR, "inventory.csv"), index=False)

def load_pending():
    return pd.read_csv(os.path.join(DATA_DIR, "pending_donations.csv"))

def save_pending(df):
    df.to_csv(os.path.join(DATA_DIR, "pending_donations.csv"), index=False)

# --- SESSION STATE ---
if "host_logged_in" not in st.session_state:
    st.session_state.host_logged_in = False
if "prediction" not in st.session_state:
    st.session_state.prediction = None

# --- SIDEBAR: GLOBAL PREDICTION INPUTS ---
st.sidebar.header("📍 Tomorrow's Conditions (Global)")
month = st.sidebar.slider("Month", 1, 12, 1)
temp = st.sidebar.number_input("Temperature (F)", value=70.0)
rain = st.sidebar.selectbox("Is it Raining?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
snow = st.sidebar.selectbox("Is it Snowing?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
holiday = st.sidebar.checkbox("Is it a Holiday?")
local_event = st.sidebar.checkbox("Is there a Local Event?")
end_of_month = st.sidebar.checkbox("Is it End of Month?")

features = np.array([[month, temp, int(rain), int(snow), int(holiday), int(local_event), int(end_of_month)]])

if st.sidebar.button("Calculate Forecast"):
    prediction = model.predict(features)[0]
    st.session_state.prediction = int(prediction)
    st.sidebar.success(f"Predicted Guests: {st.session_state.prediction}")

# --- MAIN APP ---
st.title("🍲 Soup Kitchen Management System")
st.markdown("---")

# Role Tabs
tab_user, tab_donator, tab_host = st.tabs(["👤 Guest / User", "🤝 Donator", "🔑 Host"])

# ===================== GUEST / USER TAB =====================
with tab_user:
    st.subheader("📊 Attendance Forecast")
    if st.session_state.prediction is not None:
        st.metric(label="Predicted Guests Tomorrow", value=st.session_state.prediction)
        st.info(f"**Prep Recommendation:** Prepare **{st.session_state.prediction + 5}** bowls of soup and **{st.session_state.prediction + 10}** pieces of bread.")
    else:
        st.write("Adjust sidebar conditions and click 'Calculate Forecast'.")

    st.subheader("📋 Today's Menu")
    day_choice = st.selectbox("Select Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    day_menu = menu_df[menu_df['day'] == day_choice]
    if not day_menu.empty:
        meal = day_menu.iloc[0]['meal_name']
        ingredients = day_menu.iloc[0]['ingredients'].split(", ")
        st.success(f"**Menu:** {meal}")
        st.write("**Ingredients Needed:**")
        for ing in ingredients:
            st.write(f"- {ing}")
        st.markdown("---")
        selected_ing = st.selectbox("View alternative recipes for:", ingredients)
        recipes = {
            "chicken": ["Grilled Chicken", "Chicken Fried Rice", "Chicken Salad"],
            "carrot": ["Roasted Carrots", "Carrot Soup"],
            "onion": ["Onion Soup", "Sauteed Onions"],
            "potato": ["Mashed Potatoes", "Baked Potatoes"],
            "pasta": ["Pasta Salad", "Garlic Pasta"],
            "tomato": ["Tomato Soup", "Tomato Sauce"],
        }
        if selected_ing.lower() in recipes:
            for r in recipes[selected_ing.lower()]:
                st.write(f"🍴 {r}")
        else:
            st.write("No suggestions found.")
    else:
        st.warning("No menu for this day.")

    st.subheader("📦 Current Inventory (What We Have)")
    inv = load_inventory()
    st.dataframe(inv.style.apply(lambda x: ['background-color: #ffdddd' if v < 10 else '' for v in x], subset=['quantity']))

    st.subheader("📍 Our Locations")
    locations = pd.DataFrame({
        "Location": ["Downtown Soup Kitchen", "East Side Community Center", "West End Shelter"],
        "Address": ["123 Main St, Cityville", "456 Elm Ave, Cityville", "789 Oak Rd, Cityville"],
        "Phone": ["(555) 123-4567", "(555) 987-6543", "(555) 555-1212"],
        "Email": ["downtown@soup.org", "eastside@soup.org", "westend@soup.org"],
        "Predicted Meals Tomorrow": [st.session_state.prediction or 150, st.session_state.prediction or 80, st.session_state.prediction or 120]
    })
    st.dataframe(locations)
    st.caption("Donators: Check inventory above to see what is needed most. All locations share the same stock pool for this demo.")

# ===================== DONATOR TAB =====================
with tab_donator:
    st.subheader("🤝 Make a Donation")
    st.info("Select items you want to donate. After physical delivery, the host will approve and inventory will update.")

    inv = load_inventory()
    item_list = inv["item"].tolist() + ["Other (custom)"]
    selected_item = st.selectbox("Item to Donate", item_list)
    if selected_item == "Other (custom)":
        custom_item = st.text_input("Custom Item Name")
        item_to_donate = custom_item
    else:
        item_to_donate = selected_item

    qty = st.number_input("Quantity", min_value=1, value=10)
    donor_name = st.text_input("Your Name")
    donor_email = st.text_input("Your Email")

    if st.button("Submit Donation Request"):
        if item_to_donate and donor_name and donor_email:
            pend = load_pending()
            new_id = (pend["donation_id"].max() + 1) if not pend.empty else 1
            new_row = pd.DataFrame([{
                "donation_id": new_id,
                "donor_name": donor_name,
                "email": donor_email,
                "item": item_to_donate,
                "quantity": qty,
                "status": "pending"
            }])
            pend = pd.concat([pend, new_row], ignore_index=True)
            save_pending(pend)
            st.success("Donation request submitted! Host will approve after physical check.")
        else:
            st.error("Please fill all fields.")

    st.subheader("📦 Current Inventory (See What Is Needed)")
    inv = load_inventory()
    st.dataframe(inv.style.apply(lambda x: ['background-color: #ffdddd' if v < 10 else '' for v in x], subset=['quantity']))

    st.subheader("📍 Our Locations")
    st.dataframe(locations)
    st.caption("Choose the closest location and donate what is low in stock.")

# ===================== HOST TAB =====================
with tab_host:
    if not st.session_state.host_logged_in:
        st.subheader("🔑 Host Login")
        username = st.text_input("Username", value="admin")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username == "admin" and password == "soupkitchen2025":
                st.session_state.host_logged_in = True
                st.rerun()
            else:
                st.error("Invalid credentials")
    else:
        st.success("✅ Logged in as Host")
        if st.button("Logout"):
            st.session_state.host_logged_in = False
            st.rerun()

        st.subheader("Host Dashboard")

        host_tab1, host_tab2, host_tab3 = st.tabs(["📦 Inventory", "📊 Forecast & Prep", "📋 Pending Donations"])

        with host_tab1:
            st.subheader("Current Inventory")
            inv = load_inventory()
            st.dataframe(inv.style.apply(lambda x: ['background-color: #ffdddd' if v < 10 else '' for v in x], subset=['quantity']))

        with host_tab2:
            st.subheader("Attendance Forecast")
            if st.session_state.prediction is not None:
                st.metric("Predicted Guests Tomorrow", st.session_state.prediction)
                st.info(f"Prep: {st.session_state.prediction + 5} soup bowls | {st.session_state.prediction + 10} bread pieces")
            else:
                st.write("Use sidebar to calculate forecast.")

        with host_tab3:
            st.subheader("Pending Donations to Approve")
            pend = load_pending()
            pending_df = pend[pend["status"] == "pending"]
            if pending_df.empty:
                st.info("No pending donations.")
            else:
                for idx, row in pending_df.iterrows():
                    col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                    with col1:
                        st.write(f"**{row['donor_name']}** ({row['email']})")
                    with col2:
                        st.write(f"**{row['item']}** × {row['quantity']}")
                    with col3:
                        st.write(f"ID: {row['donation_id']}")
                    with col4:
                        if st.button("✅ Approve", key=f"approve_{row['donation_id']}"):
                            # Update inventory
                            inv = load_inventory()
                            if row["item"] in inv["item"].values:
                                inv.loc[inv["item"] == row["item"], "quantity"] += row["quantity"]
                            else:
                                new_row = pd.DataFrame({"item": [row["item"]], "quantity": [row["quantity"]]})
                                inv = pd.concat([inv, new_row], ignore_index=True)
                            save_inventory(inv)

                            # Mark as approved
                            pend.loc[pend["donation_id"] == row["donation_id"], "status"] = "approved"
                            save_pending(pend)
                            st.success(f"Donation {row['donation_id']} approved and inventory updated!")
                            st.rerun()

            st.subheader("Approved Donations (History)")
            approved = pend[pend["status"] == "approved"]
            if not approved.empty:
                st.dataframe(approved)
            else:
                st.write("No approved donations yet.")

# --- FOOTER ---
st.markdown("---")
st.caption("AI-Powered Soup Kitchen Management System | Demo uses CSV files only (no database)")
import json
import os
import random
import string

USERS_FILE = os.path.join(os.path.dirname(__file__), 'users.json')

def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

def add_user(email, password):
    users = load_users()
    if email in users:
        return False  # 이미 존재
    users[email] = {'password': password}
    save_users(users)
    return True

def get_user(email):
    users = load_users()
    return users.get(email)

def update_password(email, new_password):
    users = load_users()
    if email not in users:
        return False
    users[email]['password'] = new_password
    save_users(users)
    return True

def generate_temp_password(length=8):
    chars = string.ascii_letters + string.digits
    return ''.join(random.choices(chars, k=length)) 
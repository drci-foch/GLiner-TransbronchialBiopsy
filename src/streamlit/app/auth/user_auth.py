import streamlit as st
from typing import Optional
from pathlib import Path
import json
import os

class UserAuth:
    def __init__(self):
        self.users_file = Path("users.json")
        self.users = self._load_users()
    
    def _load_users(self) -> dict:
        """Load users from file"""
        if self.users_file.exists():
            with open(self.users_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_users(self):
        """Save users to file"""
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f, indent=2)
    
    def login(self, username: str, password: str) -> bool:
        """Login user"""
        if username in self.users and self.users[username]['password'] == password:
            st.session_state.user = username
            return True
        return False
    
    def register(self, username: str, password: str, role: str = 'user') -> bool:
        """Register new user"""
        if username in self.users:
            return False
        
        self.users[username] = {
            'password': password,
            'role': role
        }
        self._save_users()
        return True
    
    def logout(self):
        """Logout user"""
        if 'user' in st.session_state:
            del st.session_state.user
    
    def get_current_user(self) -> Optional[str]:
        """Get current logged in user"""
        return st.session_state.get('user')
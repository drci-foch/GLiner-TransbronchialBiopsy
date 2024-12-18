import streamlit as st
from typing import Optional
from pathlib import Path
import json
import os
import hashlib
import secrets
from datetime import datetime

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
    
    def _hash_password(self, password: str) -> str:
        """
        Securely hash a password using SHA-256 with a salt.
        
        Args:
            password (str): The plain text password to hash
        
        Returns:
            str: A secure hash string containing salt and hashed password
        """
        # Generate a random salt
        salt = secrets.token_hex(16)  # 32 character salt
        
        # Combine password and salt, then hash
        salted_password = password + salt
        hashed_password = hashlib.sha256(salted_password.encode()).hexdigest()
        
        # Return salt and hash combined
        return f"{salt}${hashed_password}"
    
    def _verify_password(self, stored_password: str, provided_password: str) -> bool:
        """
        Verify a password against its stored hash.
        
        Args:
            stored_password (str): The stored password hash
            provided_password (str): The password to verify
        
        Returns:
            bool: True if password is correct, False otherwise
        """
        # Split stored password into salt and hash
        salt, original_hash = stored_password.split('$')
        
        # Hash the provided password with the stored salt
        salted_password = provided_password + salt
        hashed_password = hashlib.sha256(salted_password.encode()).hexdigest()
        
        # Compare hashes
        return hashed_password == original_hash
    
    def _validate_password(self, password: str) -> bool:
        """
        Validate password strength.
        
        Criteria:
        - At least 8 characters long
        - Contains at least one uppercase letter
        - Contains at least one lowercase letter
        - Contains at least one number
        - Contains at least one special character
        
        Args:
            password (str): Password to validate
        
        Returns:
            bool: True if password meets criteria, False otherwise
        """
        # Check length
        if len(password) < 8:
            return False
        
        # Check for at least one uppercase, one lowercase, one number, and one special char
        import re
        if not re.search(r'[A-Z]', password):
            return False
        if not re.search(r'[a-z]', password):
            return False
        if not re.search(r'\d', password):
            return False
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            return False
        
        return True

    def login(self, username: str, password: str) -> bool:
        """Login user"""
        if username in self.users:
            stored_password = self.users[username]['password']
            if self._verify_password(stored_password, password):
                st.session_state.user = username
                return True
        return False
   
    def register(self, username: str, password: str, role: str = 'user') -> bool:
        """Register new user"""
        # Validate password strength
        if not self._validate_password(password):
            st.error("Le mot de passe ne répond pas aux critères de sécurité.")
            return False
        
        if username in self.users:
            return False
       
        # Hash the password before storing
        hashed_password = self._hash_password(password)
        
        self.users[username] = {
            'password': hashed_password,
            'role': role,
            'created_at': datetime.now().isoformat()
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
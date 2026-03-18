import os
import bcrypt
import json
import base64
from functools import wraps
from flask import session, abort, request
from cryptography.fernet import Fernet
import smtplib
import time
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from database import get_db_connection

# Alert cooldown to prevent flooding
_last_email_sent = {} # {alert_type: timestamp}
EMAIL_COOLDOWN = 120 # 2 minutes between emails for same type
ALERT_RECIPIENT = ""

# Generate or load encryption key
# In a real environment, this should be stored in environment variables
KEY_FILE = 'secret.key'
if not os.path.exists(KEY_FILE):
    key = Fernet.generate_key()
    with open(KEY_FILE, 'wb') as f:
        f.write(key)
else:
    with open(KEY_FILE, 'rb') as f:
        key = f.read()

cipher = Fernet(key)

def hash_password(password):
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password, hashed):
    """Verify a password against a hash."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def encrypt_data(data):
    """Encrypt sensitive data (string or bytes)."""
    if isinstance(data, str):
        data = data.encode('utf-8')
    return cipher.encrypt(data).decode('utf-8')

def decrypt_data(encrypted_data):
    """Decrypt sensitive data."""
    return cipher.decrypt(encrypted_data.encode('utf-8')).decode('utf-8')

def encrypt_embedding(embedding_bytes):
    """Specialized encryption for embeddings."""
    return cipher.encrypt(embedding_bytes)

def decrypt_embedding(encrypted_embedding):
    """Specialized decryption for embeddings."""
    return cipher.decrypt(encrypted_embedding)

def log_security_event(event, user=None, details=None):
    """Log a security event to the database."""
    ip = request.remote_addr
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO security_logs (event, user, ip, details) VALUES (?, ?, ?, ?)",
        (event, user, ip, details)
    )
    conn.commit()
    conn.close()

def log_watchlist_action(action, user, case_id=None, details=None):
    """Log a watchlist action to the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO watchlist_logs (action, user, case_id, details) VALUES (?, ?, ?, ?)",
        (action, user, case_id, details)
    )
    conn.commit()
    conn.close()

def require_role(roles):
    """Decorator to require specific roles for a route."""
    if isinstance(roles, str):
        roles = [roles]
    
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'username' not in session:
                return abort(401) # Unauthorized
            
            user_role = session.get('role')
            if user_role not in roles:
                log_security_event(
                    "Unauthorized Access Attempt",
                    user=session.get('username'),
                    details=f"Attempted to access {request.path} (Role required: {roles})"
                )
                return abort(403) # Forbidden
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def validate_password_policy(password):
    """Validate password matches the security policy."""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter."
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one number."
    return True, "Success"

def is_ip_blocked(ip):
    """Check if an IP address is blocked."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM blocked_ips WHERE ip = ?", (ip,))
    blocked = cursor.fetchone() is not None
    conn.close()
    return blocked

def block_ip(ip, reason):
    """Block an IP address."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO blocked_ips (ip, reason) VALUES (?, ?)", (ip, reason))
    conn.commit()
    conn.close()
    log_security_event("IP Blocked", user="SYSTEM", details=f"IP {ip} blocked. Reason: {reason}")



def send_alert_email(subject, message, alert_type):
    """
    Queue an email alert to be sent asynchronously.
    """
    thread = threading.Thread(target=_send_email_task, args=(subject, message, alert_type), daemon=True)
    thread.start()

def _send_email_task(subject, message, alert_type):
    """
    Internal task for sending email with debouncing.
    """
    global _last_email_sent
    now = time.time()
    
    # Debounce check
    if alert_type in _last_email_sent and (now - _last_email_sent[alert_type]) < EMAIL_COOLDOWN:
        return

    # SMTP Configuration
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    sender_email = ""
    sender_password = os.environ.get("SMTP_PASS", "") 

    try:
        msg = MIMEMultipart()
        msg['From'] = f"AI Security System <{sender_email}>"
        msg['To'] = ALERT_RECIPIENT
        msg['Subject'] = f"SECURITY ALERT: {subject}"

        body = f"""
        --- SECURITY ALERT NOTIFICATION ---
        Type: {alert_type.upper()}
        Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))}
        
        Details:
        {message}
        
        Please check your security dashboard for live footage.
        """
        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
            
        _last_email_sent[alert_type] = now
        print(f"[ALERT] Email sent to {ALERT_RECIPIENT} for {alert_type}")
    except smtplib.SMTPAuthenticationError:
        print(f"[ALERT ERROR] SMTP Authentication failed for {sender_email}. Check App Password.")
    except Exception as e:
        print(f"[ALERT ERROR] Failed to send email alert ({alert_type}): {e}")

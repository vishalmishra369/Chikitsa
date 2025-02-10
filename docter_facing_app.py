from flask import Flask, render_template, request, redirect, session, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
import json
import os
import bcrypt
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app)

USERS_FILE = 'instance/users.json'
QUEUE_FILE = 'instance/queue.json'

def load_users():
    if not os.path.exists(USERS_FILE):
        os.makedirs('instance', exist_ok=True)
        with open(USERS_FILE, 'w') as f:
            json.dump([], f)
    with open(USERS_FILE, 'r') as f:
        return json.load(f)

def load_queue():
    if not os.path.exists(QUEUE_FILE):
        os.makedirs('instance', exist_ok=True)
        with open(QUEUE_FILE, 'w') as f:
            json.dump([], f)
    with open(QUEUE_FILE, 'r') as f:
        return json.load(f)

@app.route('/')
def home():
    return render_template('dindex.html')

@app.route('/patient-waiting')
def patient_waiting():
    if 'email' not in session:
        return redirect('/login')
    return render_template('patient_waiting.html')

@app.route('/doctor-dashboard')
def doctor_dashboard():
    if 'email' not in session:
        return redirect('/login')
    users = load_users()
    user = next((u for u in users if u['email'] == session['email']), None)
    if not user or user.get('role') != 'doctor':
        return redirect('/')
    queue = load_queue()
    return render_template('doctor_dashboard.html', queue=queue)

@app.route('/chat/<room>')
def chat(room):
    if 'email' not in session:
        return redirect('/login')
    return render_template('chat.html', room=room)

# Modify handle_join_queue to include patient name and update doctor dashboards
@socketio.on('join_queue')
def handle_join_queue(data=None):
    users = load_users()
    current_user = next((u for u in users if u['email'] == session['email']), None)
    
    queue = load_queue()
    patient_data = {
        'patient_email': session['email'],
        'patient_name': current_user['name'],
        'timestamp': datetime.now().isoformat(),
        'status': 'waiting'
    }
    queue.append(patient_data)
    
    with open(QUEUE_FILE, 'w') as f:
        json.dump(queue, f, indent=4)
    
    # Broadcast queue update to all connected clients
    emit('queue_update', {'queue': queue}, broadcast=True)

# Add route to accept patient
@socketio.on('accept_patient')
def accept_patient(data):
    patient_email = data['patient_email']
    doctor_email = session['email']
    
    queue = load_queue()
    for patient in queue:
        if patient['patient_email'] == patient_email:
            patient['status'] = 'accepted'
            patient['doctor_email'] = doctor_email
            
    with open(QUEUE_FILE, 'w') as f:
        json.dump(queue, f, indent=4)
    
    room = f"{patient_email}-{doctor_email}"
    emit('patient_accepted', {
        'room': room,
        'doctor_email': doctor_email
    }, room=patient_email)

@socketio.on('join')
def handle_join(data):
    room = data['room']
    join_room(room)
    emit('user_joined', {'email': session['email']}, room=room)

# Add this route for admin to create doctor accounts
@app.route('/create-doctor', methods=['GET', 'POST'])
def create_doctor():
    if 'email' not in session:
        return redirect('/login')
        
    # Check if current user is admin
    users = load_users()
    current_user = next((u for u in users if u['email'] == session['email']), None)
    if not current_user or current_user.get('role') != 'admin':
        return redirect('/')
        
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        new_doctor = {
            'name': name,
            'email': email,
            'password': hashed_password,
            'role': 'doctor',
            'created_at': datetime.now().isoformat()
        }
        
        users.append(new_doctor)
        with open(USERS_FILE, 'w') as f:
            json.dump(users, f, indent=4)
            
        return redirect('/admin-dashboard')
        
    return render_template('create_doctor.html')

# Modify the login route to redirect based on role
# Modify the login route to properly handle password verification
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        users = load_users()
        user = next((user for user in users if user['email'] == email), None)
        
        try:
            if user and bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
                session['email'] = user['email']
                
                # Redirect based on role
                if user.get('role') == 'doctor':
                    return redirect('/doctor-dashboard')
                elif user.get('role') == 'admin':
                    return redirect('/admin-dashboard')
                else:
                    return redirect('/patient-waiting')
        except ValueError:
            # Handle invalid password hash
            pass
            
        return render_template('login.html', error='Invalid credentials.')

    return render_template('login.html')
# Modify the register route to set default role as 'patient'
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        users = load_users()
        
        if any(user['email'] == email for user in users):
            return render_template('register.html', error='Email already exists.')

        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        new_user = {
            'name': name,
            'email': email,
            'password': hashed_password,
            'role': 'patient',  # Default role for self-registration
            'created_at': datetime.now().isoformat()
        }
        
        users.append(new_user)
        with open(USERS_FILE, 'w') as f:
            json.dump(users, f, indent=4)
        
        session['email'] = email
        return redirect('/questionnaire')

    return render_template('register.html')

@socketio.on('message')
def handle_message(data):
    room = data['room']
    emit('message', {
        'user': session['email'],
        'message': data['message']
    }, room=room)

# Add admin dashboard route
@app.route('/admin-dashboard')
def admin_dashboard():
    if 'email' not in session:
        return redirect('/login')
    users = load_users()
    user = next((u for u in users if u['email'] == session['email']), None)
    if not user or user.get('role') != 'admin':
        return redirect('/')
    return render_template('admin_dashboard.html', users=users)
if __name__ == '__main__':
    socketio.run(app, debug=True)

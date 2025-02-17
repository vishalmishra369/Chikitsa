from flask import Flask, render_template, request, jsonify, redirect, url_for, session,Response,flash
import json
from flask_socketio import SocketIO, emit, join_room, leave_room
import os
from pathlib import Path
import bcrypt
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
import google.generativeai as genai
from Create_modules.close_end_questionaire import get_random_close_questions
import csv
from Create_modules.open_end_questions import get_random_open_questions
import pandas as pd
from Create_modules.csv_extracter import csv_to_string
from dotenv import load_dotenv
import json
from Create_modules.image_analysis import analyze_image
from werkzeug.utils import secure_filename
from Create_modules.trained_chikitsa import chatbot_response
import cv2
import bcrypt
from datetime import datetime
from functools import wraps
# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'secret_key'

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'image_analysis/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    """
    Checks if the file has an allowed extension.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize Gemini AI
genai.configure(api_key=os.getenv("API_KEY"))

@app.route('/')
def index():
    return render_template('index.html')

# Define users.json path
USERS_FILE = "instance/users.json"

def load_users():
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)


@app.route('/questionnaire', methods=['GET', 'POST'])
def questionnaire():
    if 'email' not in session:
        return redirect('/login')
        
    if request.method == 'POST':
        age = request.form.get('age')
        gender = request.form.get('gender')
        occupation_type = request.form.get('occupation_type')
        occupation_detail = request.form.get('occupation_detail')
        
        # Determine final occupation value
        occupation = occupation_detail if occupation_type == 'Other' else occupation_type
        
        with open(USERS_FILE, 'r') as f:
            users = json.load(f)
            
        user = next((user for user in users if user['email'] == session['email']), None)
        
        if user:
            os.makedirs('instance/user_data', exist_ok=True)
            
            user_data = {
                'age': age,
                'gender': gender,
                'occupation': occupation,
                'timestamp': datetime.now().isoformat()
            }
            
            filename = f"instance/user_data/{user['name']}.json"
            with open(filename, 'w') as f:
                json.dump(user_data, f, indent=4)
                
            return redirect('/login')
            
    return render_template('questionnaire.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        os.makedirs('instance', exist_ok=True)

        if not os.path.exists(USERS_FILE):
            with open(USERS_FILE, 'w') as f:
                json.dump([], f)

        with open(USERS_FILE, 'r') as f:
            users = json.load(f)

        # Check for existing user (both email and username)
        if any(user['email'] == email or user['name'] == name for user in users):
            return render_template('register.html', error='Email or username already exists.')

        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        new_user = {
            'name': name,
            'email': email,
            'password': hashed_password,
            'role': 'user',
            'created_at': datetime.now().isoformat()
        }

        users.append(new_user)
        save_users(users)

        session['username'] = name
        session['email'] = email
        session['role'] = 'user'
        return redirect('/appointment') # Redirect to appointment page after registration

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        identifier = request.form.get('identifier')  # Using get() method for safe access
        password = request.form.get('password')

        if not identifier or not password:
            return render_template('login.html', error='Please provide all credentials')

        users = load_users()
        user = next((user for user in users if user['email'] == identifier or user['name'] == identifier), None)

        if user and bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
            session['username'] = user['name']
            session['email'] = user['email']
            session['role'] = user.get('role', 'user')

            if session['role'] == 'admin':
                return redirect('/admin/dashboard')
            elif session['role'] == 'doctor':
                return redirect('/doctor/dashboard')
            else:
                return redirect('/home') # Redirect user to appointment page after login

        return render_template('login.html', error='Invalid credentials')

    return render_template('login.html')


@app.route('/session-info')
def session_info():
    session_data = dict(session)
    return jsonify(session_data)

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/meditation')
@login_required
def meditation():
    return render_template('meditation.html')

@app.route('/log_meditation', methods=['POST'])
@login_required
def log_meditation():
    username = session.get('username')
    
    meditation_data = {
        'timestamp': datetime.now().isoformat(),
        'duration': request.json.get('duration', 300),  # Default 5 minutes
        'completed': request.json.get('completed', True)
    }
    
    # Store meditation session data
    meditation_file = f'instance/meditation_data/{username}_meditation.json'
    os.makedirs('instance/meditation_data', exist_ok=True)
    
    try:
        with open(meditation_file, 'r') as f:
            meditation_history = json.load(f)
    except FileNotFoundError:
        meditation_history = []
    
    meditation_history.append(meditation_data)
    
    with open(meditation_file, 'w') as f:
        json.dump(meditation_history, f, indent=4)
    
    return jsonify({
        'status': 'success',
        'message': 'Meditation session logged successfully',
        'total_sessions': len(meditation_history)
    })

@app.route('/get_meditation_stats')
@login_required
def get_meditation_stats():
    username = session.get('username')
    meditation_file = f'instance/meditation_data/{username}_meditation.json'
    
    try:
        with open(meditation_file, 'r') as f:
            meditation_history = json.load(f)
            
        total_minutes = sum(session['duration'] for session in meditation_history)
        total_sessions = len(meditation_history)
        
        return jsonify({
            'total_minutes': total_minutes,
            'total_sessions': total_sessions,
            'recent_sessions': meditation_history[-5:]  # Last 5 sessions
        })
    except FileNotFoundError:
        return jsonify({
            'total_minutes': 0,
            'total_sessions': 0,
            'recent_sessions': []
        })

@app.route('/personal_info')
@login_required
def personal_info():
    username = session.get('username')
    user_data_path = f'instance/user_data/{username}.json'
    
    if os.path.exists(user_data_path):
        with open(user_data_path, 'r') as f:
            user_data = json.load(f)
    else:
        user_data = {
            "age": "",
            "gender": "",
            "occupation": "",
            "timestamp": datetime.now().isoformat()
        }
    
    return render_template('personal_info.html', user_data=user_data)

@app.route('/update_personal_info', methods=['POST'])
@login_required
def update_personal_info():
    username = session.get('username')
    user_data_path = f'instance/user_data/{username}.json'
    
    updated_data = {
        "age": request.form.get('age'),
        "gender": request.form.get('gender'),
        "occupation": request.form.get('occupation'),
        "timestamp": datetime.now().isoformat()
    }
    
    os.makedirs('instance/user_data', exist_ok=True)
    with open(user_data_path, 'w') as f:
        json.dump(updated_data, f, indent=4)
    
    return redirect(url_for('personal_info'))
@app.route('/closed_ended')
def close_ended():
    random_questions = get_random_close_questions()  # Get random 5 questions
    return render_template('closed_ended.html', questions=random_questions)

@app.route('/submit_close_end', methods=['POST'])
def submit_close_ended():
    if request.method == 'POST':
        responses = []
        for question in request.form:
            answer = request.form[question]
            responses.append((question, answer))
        
        # Save responses to CSV
        save_to_csv(responses)
        
        return redirect(url_for('submit_opended'))  # Corrected here
@app.route('/mood_tracker')
@login_required
def mood_tracker():
    return render_template('mood_tracker.html')

@app.route('/log_mood', methods=['POST'])
@login_required
def log_mood():
    data = request.json
    username = session.get('username')
    mood_file = f'instance/mood_data/{username}_moods.json'
    
    os.makedirs('instance/mood_data', exist_ok=True)
    
    try:
        with open(mood_file, 'r') as f:
            moods = json.load(f)
    except FileNotFoundError:
        moods = []
    
    moods.append({
        'mood': data['mood'],
        'timestamp': data['timestamp']
    })
    
    with open(mood_file, 'w') as f:
        json.dump(moods, f)
    
    return jsonify({'status': 'success'})

@app.route('/get_moods')
@login_required
def get_moods():
    username = session.get('username')
    mood_file = f'instance/mood_data/{username}_moods.json'
    
    try:
        with open(mood_file, 'r') as f:
            moods = json.load(f)
    except FileNotFoundError:
        moods = []
    
    return jsonify({'moods': moods})
def save_to_csv(responses):
    # Get current user's email from session
    user_email = session.get('email')
    if not user_email:
        return
    
    # Load users to get username
    users = load_users()
    user = next((user for user in users if user['email'] == user_email), None)
    if not user:
        return
    
    username = user['name']
    
    # Create directory if it doesn't exist
    os.makedirs('responses/close_ended', exist_ok=True)
    
    # Define file path for this user
    file_path = f'responses/close_ended/{username}.csv'
    
    # Write responses to user's CSV file
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Question', 'Answer'])  # Write header
        for response in responses:
            writer.writerow(response)


@app.route('/open_ended', methods=['GET', 'POST'])
def submit_opended():
    if request.method == 'POST':
        responses = {key: value for key, value in request.form.items()}
        save_responses_to_csv(responses)
        return redirect(url_for('thank_you'))  # Corrected here
    else:
        random_questions = get_random_open_questions()  # Call the function from the imported file
        return render_template('open_ended.html', questions=random_questions)
    
    
def save_responses_to_csv(responses):
    # Get current user's email from session
    user_email = session.get('email')
    if not user_email:
        return
    
    # Load users to get username
    users = load_users()
    user = next((user for user in users if user['email'] == user_email), None)
    if not user:
        return
    
    username = user['name']
    
    # Create directory if it doesn't exist
    os.makedirs('responses/open_ended', exist_ok=True)
    
    # Define file path for this user
    file_path = f'responses/open_ended/{username}.csv'
    
    # Prepare the data to be saved
    data_to_save = [(question, responses[question]) for question in responses]
    
    # Create a DataFrame from the prepared data
    df = pd.DataFrame(data_to_save, columns=['Question', 'Response'])
    
    # Save to CSV
    df.to_csv(file_path, mode='w', index=False)

@app.route('/thank_you')
def thank_you():
    # Get the email from the session
    email = session.get('email')

    if email:
        # Load users from JSON file
        with open(USERS_FILE, 'r') as f:
            users = json.load(f)
        
        # Find user by email
        user = next((user for user in users if user['email'] == email), None)
        if user:
            user_name = user['name']
            # Read from user-specific files
            close_ended_str = csv_to_string(f"responses/close_ended/{user_name}.csv")
            open_ended_str = csv_to_string(f"responses/open_ended/{user_name}.csv")
        else:
            user_name = "Guest"
            close_ended_str = ""
            open_ended_str = ""
    else:
        user_name = "Guest"
        close_ended_str = ""
        open_ended_str = ""

    default = "This is my assessment of close-ended questions and open-ended questions. Please provide feedback on me."
    judge_gemini = gemini_chat(default + " " + close_ended_str + " " + open_ended_str)
    
    mainprompt = "Please summarize the following content in 150 words..."  # Your existing prompt
    summarize = gemini_chat(mainprompt + judge_gemini)
    
    return render_template('thank_you.html', judge_gemini=summarize, user_name=user_name, completejudege=judge_gemini)






@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect('/login')

from Create_modules.csv_extracter import close_ended_response , open_ended_response
genai.configure(api_key=os.getenv("API_KEY"))
defaultprompt ="""you have to act as a sexologist , gynologist , neuroloigist also health guide  
        You are a professional, highly skilled mental doctor, and health guide.
        You act as a best friend to those who talk to you , but you have to talk based on their mental health , by seeing his age intrests qualities , if you dont know ask him indirectly by asking his/her studing or any work doing currently. 
        you can use the knowlege of geeta to make the user's mind more powerfull but you dont have to give reference of krishna or arjuna etc if the user is more towards god ( hindu gods ) then u can else wont
        You can assess if someone is under mental stress by judging their communication. 
        Your goal is to make them feel happy by cracking jokes and suggesting activities that bring joy. 
        If anyone asks anything outside your field, guide them or use your knowledge to help. 
        You can speak in Hindi, Marathi, Hinglish, or any language the user is comfortable with.
        your name is chikitsa , means Cognitive Health Intelligence Knowledge with Keen Interactive Treatment Support from AI."""
prompt = "this is my assesment of close ended questions and open ended questions , so you have to talk to me accordingly "
# Create the model
generation_config = {
  "temperature": 2,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}
# username = save_current_username()
# close_ended_data = close_ended_response(username)
# open_ended_data = open_ended_response(username)

model = genai.GenerativeModel(
  model_name="gemini-1.5-pro",
  generation_config=generation_config,
  system_instruction=defaultprompt+ prompt+ " " )#+close_ended_data+open_ended_data)


chat_session = model.start_chat()

# Chat function to handle user input
def gemini_chat(user_input, history_file="dataset/intents.json"):
    try:
        # Load intents from JSON or initialize
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                intents_data = json.load(f)
        else:
            intents_data = {"intents": []}

        # Send user input to the model
        response = chat_session.send_message(user_input)

        # Create a new intent object for the conversation
        new_intent = {
            "patterns": [user_input],
            "responses": [response.text.strip()],
        }

        # Append the new intent to the intents list
        intents_data['intents'].insert(1, new_intent)

        # Save the updated intents JSON file
        with open(history_file, 'w') as f:
            json.dump(intents_data, f, indent=4)

        return response.text

    except Exception as e:
       
        response = chatbot_response(user_input)
        # Optionally log the error to a file
        with open('error.log', 'a') as log_file:
            log_file.write(f"{str(e)}\n")
        return response



def get_users_in_queue():
    users = load_users()
    queued_users = []
    
    for user in users:
        if user.get('questionnaire_completed') and not user.get('consultation_completed'):
            queued_users.append(user)
            
    return queued_users

@app.route('/admin/dashboard')
def admin_dashboard():
    # Check if user is logged in and has admin role
    if 'email' not in session:
        flash('Please login first', 'warning')
        return redirect('/login')
    
    if session.get('role') != 'admin':
        flash('Access restricted. Admin privileges required.', 'error')
        return redirect('/')
    
    # Load and pass users data to template
    users = load_users()
    current_admin = session['email']
    
    return render_template(
        'admin_dashboard.html',
        users=users,
        current_admin=current_admin,
        admin_since=get_admin_info(current_admin)
    )

def get_admin_info(admin_email):
    users = load_users()
    for user in users:
        if user['email'] == admin_email:
            return user.get('created_at', 'Unknown')
    return 'Unknown'

@app.route('/admin/create_user', methods=['GET', 'POST'])
def create_user():
    if 'email' not in session or session.get('role') != 'admin':
        return redirect('/login')
    
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        role = request.form['role']
        
        users = load_users()
        
        if any(user['email'] == email for user in users):
            return render_template('create_user.html', error='Email already exists.')
        
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        new_user = {
            'name': name,
            'email': email,
            'password': hashed_password,
            'role': role,
            'created_at': datetime.now().isoformat(),
            'created_by': session['email']
        }
        
        users.append(new_user)
        
        with open(USERS_FILE, 'w') as f:
            json.dump(users, f, indent=4)
            
        return redirect('/admin/dashboard')
        
    return render_template('create_user.html')


@app.route('/admin/update_role/<user_email>', methods=['POST'])
def update_role(user_email):
    if 'email' not in session or session.get('role') != 'admin':
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401

    # Get the new role from form data
    new_role = request.form.get('new_role')
    
    if not new_role or new_role not in ['user', 'admin', 'doctor']:
        return jsonify({'success': False, 'message': 'Invalid role'}), 400

    # Load users from JSON file
    with open(USERS_FILE, 'r') as f:
        users = json.load(f)
    
    # Update user role
    user_found = False
    for user in users:
        if user['email'] == user_email:
            user['role'] = new_role
            user['updated_at'] = datetime.now().isoformat()
            user['updated_by'] = session['email']
            user_found = True
            break
    
    if not user_found:
        return jsonify({'success': False, 'message': 'User not found'}), 404
    
    # Save updated users back to JSON file
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=4)
    
    return jsonify({'success': True, 'message': 'Role updated successfully'})

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'GET':
        return render_template('main.html')
    elif request.method == 'POST':
        user_input = request.form['user_input']
        response = gemini_chat(user_input)  # Call gemini_chat function here
        return jsonify({'response': response})
    else:
        return "Unsupported request method", 405  # Handle other methods if needed


generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

system_prompt = """
As a highly skilled medical practitioner specializing in image analysis, you are tasked with examining images for a renowned hospital. Your expertise is crucial in identifying any anomalies, diseases, or health issues that may be present in the images.
you have to analayse the image but and predict the cause or whats that 
Your Responsibilities include:

1. Detailed Analysis: Thoroughly analyze each image, focusing on identifying any abnormal findings.
2. Findings Report: Document all observed anomalies or signs of disease. Clearly articulate these findings in a structured format.
3. Recommendations and Next Steps: Based on your analysis, suggest potential next steps, including further tests or treatments as applicable.
4. Treatment Suggestions: If appropriate, recommend possible treatment options or interventions.

Important Notes:

1. Scope of Response: Only respond if the image pertains to human health issues.
2. Disclaimer: Accompany your analysis with the disclaimer: "Consult with a Doctor before making any decisions."
3. Your insights are invaluable in guiding clinical decisions. Please proceed with analysis, adhering to the structured approach outlined above.
"""
@app.route('/image_analysis', methods=['GET', 'POST'])
def image_analysis():
    analysis = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        uploaded_file = request.files['file']
        if uploaded_file.filename == '':
            return redirect(request.url)

        if uploaded_file:
            # Process the uploaded image
            image_data = uploaded_file.read()

            # Prepare image parts
            image_parts = [{
                "mime_type": "image/jpeg",
                "data": image_data
            }]
            
            # Prepare the prompt parts
            prompt_parts = [
                image_parts[0],
                system_prompt,
            ]

            # Generate a response based on the prompt and image
            response = model.generate_content(prompt_parts)
            analysis = response.text
    return render_template('image_analysis.html', analysis=analysis)


from fer import FER
# # Initialize emotion detection model
emotion_detector = FER(mtcnn=True)


# Initialize global variables
attention_status = "Not Paying Attention"
dominant_emotion = "neutral"  # Default value

def is_paying_attention(emotions_dict, threshold=0.5):
    """ Checks if the user is paying attention based on emotion scores. """
    dominant_emotion = max(emotions_dict, key=emotions_dict.get)
    emotion_score = emotions_dict[dominant_emotion]
    return emotion_score > threshold, dominant_emotion

def detect_emotion_and_attention(frame):
    """ Detects emotion and attention from the frame. """
    global attention_status, dominant_emotion

    results = emotion_detector.detect_emotions(frame)

    for result in results:
        bounding_box = result["box"]
        emotions_dict = result["emotions"]

        # Update attention status and dominant emotion
        paying_attention, dominant_emotion = is_paying_attention(emotions_dict)
        attention_status = "Paying Attention" if paying_attention else "Not Paying Attention"

        # Draw bounding box and display emotion info
        x, y, w, h = bounding_box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        emotion_text = ", ".join([f"{emotion}: {prob:.2f}" for emotion, prob in emotions_dict.items()])
        cv2.putText(frame, f"{dominant_emotion} ({attention_status})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, emotion_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return frame

# Video feed generator
def generate_frames():
    """ Captures frames from the webcam and detects emotion and attention. """
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = detect_emotion_and_attention(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Flask route for the chatbot interaction
@app.route('/talk_to_me', methods=['GET', 'POST'])
def talk_to_me():
    """ Handles the user's input and sends it to the chatbot along with emotion and attention. """
    global attention_status, dominant_emotion

    if request.method == 'GET':
        return render_template('talk_to_me.html')

    elif request.method == 'POST':
        user_input = request.form.get('user_input', '')

        # Create the prompt with the attention status and emotion
        prompt = f"The user is in a {dominant_emotion} mood and is {'paying attention' if attention_status == 'Paying Attention' else 'not paying attention'}."

        # Call the gemini_chat function with the user input and the generated prompt
        bot_response = gemini_chat(user_input + " " + prompt)

        return jsonify({'response': bot_response})

    else:
        return "Unsupported request method", 405

def log_conversation(user_input, bot_response, history_file="dataset/intents.json"):
    # Load or initialize intents data
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            intents_data = json.load(f)
    else:
        intents_data = {"intents": []}

    # Create a new intent object
    new_intent = {
        "patterns": [user_input],
        "responses": [bot_response],
    }

    # Append the new intent
    intents_data['intents'].append(new_intent)

    # Save the updated intents JSON file
    with open(history_file, 'w') as f:
        json.dump(intents_data, f, indent=4)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == "__main__":
    app.run(debug=True)
    
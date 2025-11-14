import io
import os
import base64
import pickle
from datetime import datetime, date

from flask import (Flask, render_template, request, redirect, url_for,
                   flash, jsonify)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (LoginManager, login_user, login_required,
                         logout_user, current_user, UserMixin)
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image
import numpy as np
import face_recognition

app = Flask(__name__)
app.config['SECRET_KEY'] = 'replace-with-a-secure-random-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///attendance.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

# ----------------- MODELS -----------------
class Faculty(UserMixin, db.Model):
    __tablename__ = "faculty"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(300), nullable=False)
    full_name = db.Column(db.String(200))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Student(db.Model):
    __tablename__ = "student"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    roll = db.Column(db.String(100), unique=True, nullable=False)
    branch = db.Column(db.String(100), nullable=False)
    year = db.Column(db.String(20), nullable=False)
    # store face encoding as pickled blob
    face_encoding = db.Column(db.LargeBinary, nullable=False)  # pickled numpy array
    added_on = db.Column(db.DateTime, default=datetime.utcnow)


class Attendance(db.Model):
    __tablename__ = "attendance"
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False)
    student_name = db.Column(db.String(200), nullable=False)
    roll = db.Column(db.String(100), nullable=False)
    branch = db.Column(db.String(100), nullable=False)
    year = db.Column(db.String(20), nullable=False)
    date = db.Column(db.Date, nullable=False)
    time = db.Column(db.String(20), nullable=False)
    marked_on = db.Column(db.DateTime, default=datetime.utcnow)


# ----------------- LOGIN -----------------
@login_manager.user_loader
def load_user(user_id):
    return Faculty.query.get(int(user_id))


# ----------------- ROUTES -----------------
@app.route('/')
def index():
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()
        fullname = request.form.get('full_name', '').strip()

        if Faculty.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
            return redirect(url_for('register'))

        user = Faculty(username=username, full_name=fullname)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        flash('Registration successful. Please login.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()
        user = Faculty.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid username or password', 'danger')
        return redirect(url_for('login'))
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

from datetime import date

@app.route("/dashboard")
@login_required
def dashboard():
    today = date.today()

    # 1. Get all branches and years
    classes = db.session.query(Student.branch, Student.year).distinct().all()

    summary = []

    for branch, year in classes:
        # Total students in class
        total_students = Student.query.filter_by(branch=branch, year=year).count()

        # Total present today
        present_count = Attendance.query.filter_by(
            date=today, branch=branch, year=year
        ).count()

        # Absent = total - present
        absent_count = total_students - present_count

        summary.append({
            "branch": branch,
            "year": year,
            "present": present_count,
            "absent": absent_count
        })

    return render_template("dashboard.html", summary=summary, today=today)



# ------------- ADD STUDENT -------------
@app.route('/add-student', methods=['GET', 'POST'])
@login_required
def add_student():
    branches = ['CSE', 'IT', 'ME', 'CE', 'ECE', 'EE']  # you can customize
    years = ['1st', '2nd', '3rd', '4th']
    if request.method == 'POST':
        name = request.form['name'].strip()
        roll = request.form['roll'].strip()
        branch = request.form['branch']
        year = request.form['year']
        file = request.files.get('image')

        if not (name and roll and branch and year and file):
            flash('All fields are required', 'danger')
            return redirect(url_for('add_student'))

        # read uploaded image and compute face encoding
        img_stream = file.read()
        pil_img = Image.open(io.BytesIO(img_stream)).convert('RGB')
        np_img = np.array(pil_img)
        boxes = face_recognition.face_locations(np_img)

        if len(boxes) == 0:
            flash('No face detected in the uploaded image. Please upload a clear face image.', 'danger')
            return redirect(url_for('add_student'))
        # compute encoding (use first face)
        encs = face_recognition.face_encodings(np_img, boxes)
        encoding = encs[0]

        # pickle the numpy array
        pickled = pickle.dumps(encoding)

        # create student
        if Student.query.filter_by(roll=roll).first():
            flash('Student with this roll already exists', 'danger')
            return redirect(url_for('add_student'))

        student = Student(name=name, roll=roll, branch=branch, year=year, face_encoding=pickled)
        db.session.add(student)
        db.session.commit()
        flash('Student added successfully', 'success')
        return redirect(url_for('dashboard'))

    return render_template('add_student.html', branches=branches, years=years)


# ------------- MARK ATTENDANCE PAGE (frontend) -------------
@app.route('/mark-attendance')
@login_required
def mark_attendance():
    return render_template('mark_attendance.html')


# ------------- RECOGNITION ENDPOINT -------------
@app.route('/recognize', methods=['POST'])
@login_required
def recognize():
    """
    Expects JSON: {"image": "data:image/jpeg;base64,...."}
    Returns JSON with recognized student info (if any) and whether attendance was marked.
    """
    data = request.json
    img_b64 = data.get('image', '')
    if not img_b64:
        return jsonify({"status": "error", "message": "No image provided"}), 400

    # strip header if present
    if ',' in img_b64:
        img_b64 = img_b64.split(',', 1)[1]

    try:
        img_bytes = base64.b64decode(img_b64)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        np_img = np.array(pil_img)
    except Exception as e:
        return jsonify({"status": "error", "message": "Invalid image data"}), 400

    # find face locations & encodings in frame
    face_locations = face_recognition.face_locations(np_img)
    if len(face_locations) == 0:
        return jsonify({"status": "ok", "found": False})  # no face in this frame

    encodings = face_recognition.face_encodings(np_img, face_locations)

    # load all students' encodings
    students = Student.query.all()
    known_encodings = []
    student_map = []  # parallel list of Student
    for s in students:
        try:
            enc = pickle.loads(s.face_encoding)
        except Exception:
            continue
        known_encodings.append(enc)
        student_map.append(s)

    # For each detected face in the frame, try to match
    results = []
    for enc in encodings:
        if len(known_encodings) == 0:
            continue
        distances = face_recognition.face_distance(known_encodings, enc)
        best_idx = int(np.argmin(distances))
        best_distance = float(distances[best_idx])
        # threshold: 0.5 is typical; tune for your environment. Lower -> stricter
        if best_distance <= 0.5:
            matched = student_map[best_idx]
            # check if attendance already marked today
            today = date.today()
            existing = Attendance.query.filter_by(roll=matched.roll, date=today).first()
            if existing:
                results.append({
                    "student_id": matched.id,
                    "name": matched.name,
                    "roll": matched.roll,
                    "branch": matched.branch,
                    "year": matched.year,
                    "marked": False,
                    "reason": "already_marked"
                })
            else:
                # mark attendance
                now_time = datetime.now().strftime("%H:%M:%S")
                att = Attendance(
                    student_id=matched.id,
                    student_name=matched.name,
                    roll=matched.roll,
                    branch=matched.branch,
                    year=matched.year,
                    date=today,
                    time=now_time
                )
                db.session.add(att)
                db.session.commit()
                results.append({
                    "student_id": matched.id,
                    "name": matched.name,
                    "roll": matched.roll,
                    "branch": matched.branch,
                    "year": matched.year,
                    "marked": True
                })
        else:
            # no match
            results.append({"matched": False, "distance": best_distance})

    return jsonify({"status": "ok", "found": True, "results": results})

@app.route("/attendance_dates")
@login_required
def attendance_dates():
    dates = db.session.query(Attendance.date).distinct().all()
    return render_template("attendance_dates.html", dates=dates)

@app.route("/attendance_filter/<string:date>", methods=["GET", "POST"])
@login_required
def attendance_filter(date):
    if request.method == "POST":
        branch = request.form.get("branch")
        year = request.form.get("year")
        return redirect(url_for("attendance_list", date=date, branch=branch, year=year))

    return render_template("attendance_filter.html", date=date)

@app.route("/attendance_list/<string:date>/<string:branch>/<string:year>")
@login_required
def attendance_list(date, branch, year):
    # Convert date string back to Python date
    from datetime import datetime
    selected_date = datetime.strptime(date, "%Y-%m-%d").date()

    # 1. Get present students
    present_students = Attendance.query.filter_by(
        date=selected_date,
        branch=branch,
        year=year
    ).all()

    # 2. Get all students of that branch+year
    all_students = Student.query.filter_by(branch=branch, year=year).all()

    # 3. Identify ABSENT students
    present_rolls = {p.roll for p in present_students}

    absent_students = [s for s in all_students if s.roll not in present_rolls]

    return render_template(
        "attendance_list.html",
        date=selected_date,
        branch=branch,
        year=year,
        present_students=present_students,
        absent_students=absent_students
    )


# ------------- UTILS -------------
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


if __name__ == '__main__':
    if not os.path.exists('attendance.db'):
        with app.app_context():
            db.create_all()
    app.run(host='0.0.0.0', port=5000, debug=True)

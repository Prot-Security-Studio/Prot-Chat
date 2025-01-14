from flask import Flask, render_template, request, redirect, session, url_for, jsonify
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import os


app = Flask(__name__)
app.secret_key = 'uftayufsayfyygufwftfxc'


# 数据库设置相关函数
def create_tables():
    conn = sqlite3.connect('user_database.db')
    cursor = conn.cursor()
    # 创建用户表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            face_images_paths TEXT,
            friend_list TEXT,
            face_login_enabled INTEGER DEFAULT 0,
            theme TEXT DEFAULT 'light'
        )
    ''')
    # 创建消息表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender_id INTEGER,
            receiver_id INTEGER,
            message_text TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()


# 人脸录入和识别相关函数
# 加载Haar级联分类器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def load_saved_faces(recorder_name):
    saved_faces_path = f'face_images/{recorder_name}_faces.npy'
    if os.path.exists(saved_faces_path):
        return np.load(saved_faces_path)
    else:
        return None


def face_recognition(recorder_name, uploaded_image_path):
    saved_faces = load_saved_faces(recorder_name)
    if saved_faces is None:
        return False
    uploaded_image = cv2.imread(uploaded_image_path)
    if uploaded_image is None:
        return False
    gray_uploaded = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2GRAY)
    faces_uploaded = face_cascade.detectMultiScale(gray_uploaded, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                           flags=cv2.CASCADE_SCALE_IMAGE)
    if len(faces_uploaded) > 0:
        (x, y, w, h) = faces_uploaded[0]
        detected_face_image = gray_uploaded[y:y + h, x:x + w]
        detected_face_image = cv2.resize(detected_face_image, saved_faces[0].shape[:2])

        # 计算检测到的人脸图像的LBP特征
        lbp_detect = local_binary_pattern(detected_face_image, 8, 1, method='uniform')
        lbp_detect_hist, _ = np.histogram(lbp_detect.ravel(), bins=np.arange(0, 59), range=(0, 58))
        lbp_detect_hist = lbp_detect_hist.astype("float")
        lbp_detect_hist /= (lbp_detect_hist.sum() + 1e-7)

        for saved_face in saved_faces:
            # 计算保存的人脸图像的LBP特征
            lbp_saved = local_binary_pattern(saved_face, 8, 1, method='uniform')
            lbp_saved_hist, _ = np.histogram(lbp_saved.ravel(), bins=np.arange(0, 59), range=(0, 58))
            lbp_saved_hist = lbp_saved_hist.astype("float")
            lbp_saved_hist /= (lbp_saved_hist.sum() + 1e-7)

            # 计算特征之间的距离（这里使用卡方距离）
            chi_squared_distance = 0.5 * np.sum(
                [(a - b) ** 2 / (a + b + 1e-10) for (a, b) in zip(lbp_detect_hist, lbp_saved_hist)])
            if chi_squared_distance < 0.3:  # 设置一个相似度阈值，可根据实际情况调整
                return True
    return False


create_tables()


@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('chat_room'))
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        login_method = request.form.get('login_method')

        if not username:
            return "用户名不能为空。"

        if login_method == 'password':
            if not password:
                return "密码不能为空。"
            conn = sqlite3.connect('user_database.db')
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE username =?', (username,))
            user = cursor.fetchone()
            if user:
                if check_password_hash(user[2], password):
                    session['user_id'] = user[0]
                    return redirect(url_for('chat_room'))
                else:
                    return "密码错误，请重试。"
            else:
                return "用户名不存在，请注册。"
        elif login_method == 'face':
            uploaded_image = request.files['face_image']
            if uploaded_image and uploaded_image.filename.endswith(('.jpg', '.png')):
                uploaded_image_path = os.path.join('uploads', uploaded_image.filename)
                uploaded_image.save(uploaded_image_path)
                if face_recognition(username, uploaded_image_path):
                    conn = sqlite3.connect('user_database.db')
                    cursor = conn.cursor()
                    cursor.execute('SELECT * FROM users WHERE username =?', (username,))
                    user = cursor.fetchone()
                    if user:
                        session['user_id'] = user[0]
                        return redirect(url_for('chat_room'))
                    else:
                        return "用户信息获取失败，请重试。或者前往sdrj.imwzr.top/register   重新注册"
                else:
                    return "人脸登录失败，请重试。"
            else:
                return "请上传有效的人脸图片（jpg或png格式）。"
    return render_template('login.html')


@app.route('/chat_room')
def chat_room():
    if 'user_id' in session:
        conn = sqlite3.connect('user_database.db')
        cursor = conn.cursor()
        cursor.execute('SELECT username, friend_list, theme FROM users WHERE id =?', (session['user_id'],))
        user_info = cursor.fetchone()
        if user_info:
            username = user_info[0]
            friend_list = user_info[1].split(',') if user_info[1] else []
            theme = user_info[2]
            conn.close()
            return render_template('chat_room.html', username = username, friend_list = friend_list, theme = theme, user_id = session['user_id'])
        else:
            conn.close()
            return "用户信息获取失败，请重试。或者前往sdrj.imwzr.top/register   重新注册"
    return redirect(url_for('index'))


@app.route('/add_friend', methods=['GET', 'POST'])
def add_friend():
    if 'user_id' in session:
        if request.method == 'POST':
            friend_id = request.form['friend_id']
            conn = sqlite3.connect('user_database.db')
            cursor = conn.cursor()
            cursor.execute('SELECT friend_list FROM users WHERE id =?', (session['user_id'],))
            result = cursor.fetchone()
            if result:  # 先检查 fetchone() 的结果是否为 None
                current_friend_list = result[0].split(',') if result[0] else []
                if friend_id not in current_friend_list:
                    cursor.execute('UPDATE users SET friend_list =? WHERE id =?',
                                  (','.join(current_friend_list + [friend_id]), session['user_id']))
                    conn.commit()
            conn.close()
            return redirect(url_for('chat_room'))
        return render_template('add_friend.html')
    return redirect(url_for('index'))


@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if 'user_id' in session:
        if request.method == 'POST':
            theme = request.form['theme']
            face_login_enabled = request.get('face_login_enabled') == 'on'
            conn = sqlite3.connect('user_database.db')
            cursor = conn.cursor()
            cursor.execute('UPDATE users SET face_login_enabled =?, theme =? WHERE id =?',
                          (face_login_enabled, theme, session['user_id']))
            conn.commit()
            conn.close()
            return redirect(url_for('chat_room'))
        return render_template('settings.html')
    return redirect(url_for('index'))


@app.route('/send_message', methods=['POST'])
def send_message():
    # 检查用户是否登录
    if 'user_id' in session:
        data = request.get_json()
        message = data['message']
        receiver_id = data.get('receiver_id')  # 获取接收者的用户 ID
        # 将消息保存到数据库
        conn = sqlite3.connect('user_database.db')
        cursor = conn.cursor()
        cursor.execute('INSERT INTO messages (sender_id, receiver_id, message_text) VALUES (?,?,?)',
                      (session['user_id'], receiver_id, message))
        conn.commit()
        conn.close()
        return jsonify({"status": "success"})
    else:
        return jsonify({"status": "error", "message": "用户未登录"})


@app.route('/get_new_messages', methods=['GET'])
def get_new_messages():
    if 'user_id' in session:
        conn = sqlite3.connect('user_database.db')
        cursor = conn.cursor()
        cursor.execute('SELECT sender_id, message_text, timestamp FROM messages WHERE receiver_id IS NULL OR receiver_id =?',
                      (session['user_id'],))
        messages = cursor.fetchall()
        # 将获取到的消息转换为合适的格式以便返回给前端
        new_messages = []
        for message in messages:
            sender_id, message_text, timestamp = message
            new_messages.append({
                "sender_id": sender_id,
                "message_text": message_text,
                "timestamp": timestamp
            })
        conn.close()
        return jsonify(new_messages)
    else:
        return jsonify({"status": "error", "message": "用户未登录"})


@app.route('/send_private_message', methods=['POST'])
def send_private_message():
    if 'user_id' in session:
        data = request.get_json()
        receiver_id = data.get('receiver_id')
        message = data.get('message')
        if receiver_id and message:
            conn = sqlite3.connect('user_database.db')
            cursor = conn.cursor()
            cursor.execute('INSERT INTO messages (sender_id, receiver_id, message_text) VALUES (?,?,?)',
                          (session['user_id'], receiver_id, message))
            conn.commit()
            conn.close()
            return jsonify({"status": "success"})
        else:
            return jsonify({"status": "error", "message": "接收者ID或消息内容不能为空"})
    else:
        return jsonify({"status": "error", "message": "用户未登录"})


@app.route('/get_private_messages', methods=['GET'])
def get_private_messages():
    if 'user_id' in session:
        conn = sqlite3.connect('user_database.db')
        cursor = conn.cursor()
        cursor.execute('SELECT sender_id, receiver_id, message_text, timestamp FROM messages WHERE (sender_id =? AND receiver_id =?) OR (sender_id =? AND receiver_id =?)',
                      (session['user_id'], request.args.get('friend_id'), request.args.get('friend_id'), session['user_id']))
        messages = cursor.fetchall()
        private_messages = []
        for message in messages:
            sender_id, receiver_id, message_text, timestamp = message
            private_messages.append({
                "sender_id": sender_id,
                "receiver_id": receiver_id,
                "message_text": message_text,
                "timestamp": timestamp
            })
        conn.close()
        return jsonify(private_messages)
    else:
        return jsonify({"status": "error", "message": "用户未登录"})


@app.route('/private_chat/<friend_id>')
def private_chat(friend_id):
    if 'user_id' in session:
        return render_template('private_chat.html', friend_id=friend_id)
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=91)
from flask import Flask, render_template, request, jsonify
import datetime
import re
import os
import google.generativeai as genai

# Set your Gemini API key (replace or store in environment variable)
genai.configure(api_key=os.getenv("AIzaSyDbmTcnEmNRCnXV6pX51WrrusyIAvhkJKc") or "AIzaSyDbmTcnEmNRCnXV6pX51WrrusyIAvhkJKc")

app = Flask(__name__)

# models = genai.list_models()
# for m in models:
#     print(m.name)


def process_booking_query(message):

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        chat = model.start_chat()
        response = chat.send_message(
            f"""You are a helpful booking assistant.
            if user asks questions other than booking tell them to ask about booking.
            Keep the tone friendly and clear but well formatted.
            Avoid unnecessary details. User asked: '{message}'"""
        )
        return response.text
    except Exception as e:
        return f"Error with Gemini API: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    response = process_booking_query(user_message)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
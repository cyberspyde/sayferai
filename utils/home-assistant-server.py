from flask import Flask, request, abort
import json
from my_initializer import synthesize_once
from sayferai_telegrambot import send_emergency_message

app = Flask(__name__)

@app.route('/activate_emergency', methods=['POST'])
def activate_emergency():
    if request.method == 'POST':
        print(request.json)
        synthesize_once("Favqulotda holat aktivlashtirildi, uyni egalariga xabar berilmoqda")
        send_emergency_message("Favqulotda holat aktivlashtirildi")
        return 'Test Success', 200
    else:
        abort(400)

@app.route('/deactivate_emergency', methods=['POST'])
def deactivate_emergency():
    if request.method == 'POST':
        print(request.json)
        synthesize_once("Favqulotda holat o'chirildi")
        send_emergency_message("Favqulotda holat o'chirildi")
        return 'Test Success', 200
    else:
        abort(400)

@app.route('/open_door', methods=['POST'])
def open_door():
    if request.method == 'POST':
        print(request.json)
        synthesize_once("eshiklar ochildi")
        return 'Test Success', 200
    else:
        abort(400)

@app.route('/close_door', methods=['POST'])
def close_door():
    if request.method == 'POST':
        print(request.json)
        synthesize_once("eshiklar yopildi")
        return 'Test Success', 200
    else:
        abort(400)
        
@app.route('/phone_battery_low', methods=['POST'])
def phone_battery_low():
    if request.method == 'POST':
        print(request.json)
        synthesize_once("Telefoningizni zaryadi kam qoldi, zaryadkaga qo'yishni unutmang, ertaga muhim kun.")
        return 'Test Success', 200
    else:
        abort(400)

app.run('0.0.0.0', '5000')

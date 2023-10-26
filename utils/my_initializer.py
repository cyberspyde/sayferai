import numpy as np
from nltk.stem.porter import PorterStemmer
import pyaudio, torch, string, random, json, subprocess, wikipedia, os, requests, threading, uuid, nltk, re, threading
from subprocess import call, Popen
from playsound import playsound
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from transformers import pipeline
from transformers import RobertaTokenizer, RobertaForQuestionAnswering
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import sounddevice as sd
from utils.text2num import text2num
from utils.num2text import num2text

date = datetime.today().strftime("%y:%m:%d, %H:%M:%S")
whisper_url = "https://cyberspyde-whisper-uz-api.hf.space/transcribe"

BASE_DIR = os.path.dirname(os.path.abspath(os.path.join(__file__, '..')))
SETTINGS_FILE = os.path.join(BASE_DIR, 'assets/settings.conf')
KNOWLEDGE_FILES = os.path.join(BASE_DIR, 'testing/Knowledge Base/')
SPORTS_KNOWLEDGE_FILE = os.path.join(KNOWLEDGE_FILES, 'sports-knowledge.json')
TECHNOLOGY_KNOWLEDGE_FILE = os.path.join(KNOWLEDGE_FILES, 'technology-knowledge.json')
POLITICS_KNOWLEDGE_FILE = os.path.join(KNOWLEDGE_FILES, 'politics-knowledge.json')
SOCIETY_KNOWLEDGE_FILE = os.path.join(KNOWLEDGE_FILES, 'society-knowledge.json')
CULTURE_KNOWLEDGE_FILE = os.path.join(KNOWLEDGE_FILES, 'culture-knowledge.json')
BUSINESS_KNOWLEDGE_FILE = os.path.join(KNOWLEDGE_FILES, 'business-knowledge.json')
FACTUAL_KNOWLEDGE_FILE = os.path.join(KNOWLEDGE_FILES, 'factual-knowledge.json')
ANALYTICAL_KNOWLEDGE_FILE = os.path.join(KNOWLEDGE_FILES, 'analytical-knowledge.json')
SUBJECTIVE_KNOWLEDGE_FILE = os.path.join(KNOWLEDGE_FILES, 'subjective-knowledge.json')
OBJECTIVE_KNOWLEDGE_FILE = os.path.join(KNOWLEDGE_FILES, 'objective-knowledge.json')
QURAN_FILE = os.path.join(BASE_DIR, 'assets/Religion/Quran-latin.json')
CACHE_DIR = os.path.join(BASE_DIR, '.cache/huggingface/transformers/')

# Load settings from a configuration file
with open(SETTINGS_FILE, 'r') as f:
    settings = json.load(f)

voice_activation = settings['voice_activation']
robot_name = settings['robot_name'].lower()
gpt3 = settings['gpt3']
robertaqna_settings = settings['robertaqna']
#logging = settings['logging']

# Load knowledge bases from JSON files
with open(SPORTS_KNOWLEDGE_FILE, 'r') as f:
    sport_knowledge = json.load(f)

with open(TECHNOLOGY_KNOWLEDGE_FILE, 'r') as f:
    technology_knowledge = json.load(f)

with open(POLITICS_KNOWLEDGE_FILE, 'r') as f:
    politics_knowledge = json.load(f)

with open(SOCIETY_KNOWLEDGE_FILE, 'r') as f:
    society_knowledge = json.load(f)

with open(CULTURE_KNOWLEDGE_FILE, 'r') as f:
    culture_knowledge = json.load(f)

with open(BUSINESS_KNOWLEDGE_FILE, 'r') as f:
    business_knowledge = json.load(f)

with open(FACTUAL_KNOWLEDGE_FILE, 'r') as f:
    factual_knowledge = json.load(f)

with open(ANALYTICAL_KNOWLEDGE_FILE, 'r') as f:
    analytical_knowledge = json.load(f)

with open(SUBJECTIVE_KNOWLEDGE_FILE, 'r') as f:
    subjective_knowledge = json.load(f)

with open(OBJECTIVE_KNOWLEDGE_FILE, 'r') as f:
    objective_knowledge = json.load(f)

with open(QURAN_FILE, 'r') as f:
    quranData = json.load(f)

if robertaqna_settings == True:
    roberta_qna_tokenizer = RobertaTokenizer.from_pretrained("deepset/roberta-base-squad2", cache_dir=CACHE_DIR)
    roberta_qna_model = RobertaForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2", cache_dir=CACHE_DIR)
    
wikipedia.set_lang("uz")

day_first = {
    1 : "O'n",
    2 : "Yigirma",
    3 : "O'ttiz"}
day_second = {
    1 : "Birinchi",
    2 : "Ikkinchi",
    3 : "Uchinchi",
    4 : "To'rtinchi",
    5 : "Beshinchi",
    6 : "Oltinchi",
    7 : "Yettinchi",
    8 : "Sakkizinchi",
    9 : "To'qqizinchi",
    0 : "inchi"}
year_first = {
        1 : "bir ming",
        2 : "ikki ming",
        3 : "uch ming",
        4 : "to'rt ming",
        5 : "besh ming",
        6 : "olti ming",
        7 : "yetti ming",
        8 : "sakkiz ming",
        9 : "to'qqiz ming"
    }
year_second = {
        1 : "bir yuz",
        2 : "ikkiyuz",
        3 : "uchyuz",
        4 : "to'rtyuz",
        5 : "beshyuz",
        6 : "oltiyuz",
        7 : "yettiyuz",
        8 : "sakkizyuz",
        9 : "to'qqizyuz",
        0 : ""
    }
year_third = {
        1 : "o'n",
        2 : "yigirma",
        3 : "o'ttiz",
        4 : "qirq",
        5 : "ellik",
        6 : "oltmish",
        7 : "yetmish",
        8 : "sakson",
        9 : "to'qson",
        0 : ""

    }
year_fourth = {
        1 : "birinchi",
        2 : "ikkinchi",
        3 : "uchinchi",
        4 : "to'rtinchi",
        5 : "beshinchi",
        6 : "oltinchi",
        7 : "yettinchi",
        8 : "sakkizinchi",
        9 : "to'qqizzinchi",
        0 : "inchi"
    }

USE_ONNX = False # change this to True if you want to test onnx model
silero_vad_path = r'c:\\Users\\ilhom\\.cache\\torch\\hub\\snakers4_silero-vad_master'
#torch.set_num_threads(1)
#snakers4/silero-vad
vad_model, vad_utils = torch.hub.load(silero_vad_path,
                              model='silero_vad',
                              source='local',
                              force_reload=True,
                              onnx=USE_ONNX)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained("GitNazarov/whisper-small-pt-3-uz")
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained("GitNazarov/whisper-small-pt-3-uz")
#whisper_model.to(device)


#Initialization for STT model
def recognize_once():
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    STT_SAMPLE_RATE = 16000
    CHUNK = int(STT_SAMPLE_RATE / 10)
    audio = pyaudio.PyAudio()
    voiced_confidences = []

    recognize_vad_stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=STT_SAMPLE_RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
    frames = []
    record_duration = 30
    voiced_confidences = []
    print("start recording...")
    talked_once = False
    for i in range(int(STT_SAMPLE_RATE / CHUNK * record_duration)):
        audio_chunk = recognize_vad_stream.read(CHUNK)
        frames.append(audio_chunk)
        audio_int16 = np.frombuffer(audio_chunk, np.int16)
        audio_float32 = int2float(audio_int16)
        new_confidence = vad_model(torch.from_numpy(audio_float32), 16000).item()
        voiced_confidences.append(new_confidence)    

        if np.average(voiced_confidences[-5:]) > 0.5:
            talked_once = True

        if talked_once == True and len(voiced_confidences) > int(STT_SAMPLE_RATE / CHUNK * 3) and np.average(voiced_confidences[-5:]) < 0.5:
            print("silence is detected, passing the audio chunk to the transcriber")
            break
    print("recording stopped.")
    return frames


silero_model_path = r'c:\\Users\\ilhom\\.cache\\torch\\hub\\snakers4_silero-models_master'
#snakers4/silero-models
tts_model, exampletext = torch.hub.load(repo_or_dir=silero_model_path,
                                    model='silero_tts',
                                    source='local',
                                    language='uz',
                                    speaker='v4_uz')
sample_rate = 24000
speaker = 'dilnavoz'
# put_accent=True
# put_yo=True

#<prosody rate="x-slow">man sekinroq gapiraman</prosody>, <break time="2000ms"/><prosody rate="fast">tez gapirishim mumkin.</prosody>
#<prosody pitch="x-low">teskarisi, past tonda gapiraman</prosody><prosody pitch="x-high"> yuqori tonda gapirishim mumkin </prosody>

def synthesize_once(mytext):
    exampletext = """
        <speak>
        <p>
            {var}
        </p>
        </speak>
        """.format(var=mytext)
    tts_audio = tts_model.apply_tts(ssml_text=exampletext,
                        speaker=speaker,
                        sample_rate=sample_rate)
    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=sample_rate,
                    output=True)

    audio_array = tts_audio.numpy()
    audio_bytes = audio_array.astype(np.float32).tobytes()

    stream.write(audio_bytes)

    stream.stop_stream()
    stream.close()
    p.terminate()

def Logging(input):
    if not input.strip() == '' and len(input.strip()) > 2:
        with open('assets/Logs/log_file.json', 'r+') as log_file:
            data = json.load(log_file)

            log = data['log']
            log.append(input.strip())
            date_array = data['date']
            date_array.append(date)

            data['log'] = log
            data['date'] = date_array
            print(data)
            log_file.seek(0)
            json.dump(data, log_file, indent=4)
    else:
        print('input invalid, logging skipped')
    # logs.append(log)
    # dates.append(date)


    # new_log_data = {"log" : logs, "date" : dates}
    # log_file.write(json.dumps(new_log_data))
    # log_file.close()

    # rightnow = datetime.today().strftime("%y:%m:%d, %H:%M:%S")
    # minute = 40
    # index_numbers = []
    # for i in dates:
    #     myobj = datetime.strptime(i, "%y:%m:%d, %H:%M:%S")
    #     minute = myobj.strftime("%M")
    #     if minute == "46":
    #         print(myobj)
    #         index_numbers.append(dates.index(i))
    # print(index_numbers)

def todays_Logs(input):
    if (get_response(input) == "bugungi kiritilgan barcha ma`lumotlar o`qib eshittiraman"):
        with open('assets/Logs/log_file.json', 'r') as log_file:
            data = json.load(log_file)
            
            logs = data['log']
            dates = data['date']
            dates_formatted = []
        
            for t in dates:
                dates_formatted.append(datetime.strptime(t, "%y:%m:%d, %H:%M:%S"))
            
            
            today = datetime.today().strftime("%y:%m:%d")
            days_in_logs = []

            for k in dates_formatted:
                days_in_logs.append(k.strftime("%y:%m:%d"))

            for i, e in enumerate(days_in_logs):
                if e == today:
                    speech_synthesizer.speak_text_async(logs[i]).get()

#todays_Logs("bugungi kiritilgan ma'lumotlarni aytib ber")

def suraExtractor(input):
    numbers = re.findall('[0-9]+', input)

    try:
        sura = numbers[0]
    except IndexError:
        sura = 0
    return int(sura)

def ayaExtractor(input):
    numbers = re.findall('[0-9]+', input)
    
    try:
        aya = numbers[1]
    except IndexError:
        aya = 0
    return int(aya)

def askQuranInUzbek(input):
    #sura = suraExtractor(input)
    aya = 0
    sura = input
    try:
        if sura > 114 or sura < 1:
            print("Bunday sura mavjud emas")
        else:
            playsound(f"assets\\Audio\\Quran\\{sura}.mp3")

        print(f"Sura - {sura}, Aya - {aya}")

        for a in range(len(quranData)):
            if quranData[a]['sura'] == sura and quranData[a]['aya'] == aya:
                    result = ''.join([i for i in quranData[a]['translation'] if not i.isdigit()])
                    synthesize_once(result)
            elif aya == 0 and quranData[a]['sura'] == sura:
                result = ''.join([i for i in quranData[a]['translation'] if not i.isdigit()])
                synthesize_once(result)
    except Exception as e:
        print(e)

def validate(model,
             inputs: torch.Tensor):
    with torch.no_grad():
        outs = model(inputs)
    return outs

def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1/32768
    sound = sound.squeeze()  # depends on the use case
    return sound

def get_current_time():
    now = datetime.now()
    hour = now.hour
    minute = now.minute

    if hour > 12:
        hour -= 12

    hour_say = num2text(hour)
    minute_say = num2text(minute)

    if minute == 0:
        minute_say = ""

    current_time = f"Hozir soat {hour_say}dan {minute_say} daqiqa o'tdi."

    return current_time

def day_filter(answer):
    answer = drop_characters(answer)

    digits = [int(s) for s in answer.split() if s.isdigit() and int(s) <= 31 and int(s) >= 10]
    digits2 = [int(s) for s in answer.split() if s.isdigit() and int(s) <= 9]

    digits_in_text = []
    digits_in_text2 = []

    for digit in digits:
        sp = [int(a) for a in str(digit)]
        digits_in_text.append(sp)

    for digit in digits2:
        sp = [int(a) for a in str(digit)]
        digits_in_text2.append(sp)

    day_first_values = []
    day_second_values = []

    day_first_values2 = []

    for p in digits_in_text:
        day_first_values.append(str(p[0]).replace(str(p[0]), day_first[p[0]]))
        day_second_values.append(str(p[1]).replace(str(p[1]), day_second[p[1]]))


    for p in digits_in_text2:
        day_first_values2.append(str(p[0]).replace(str(p[0]), day_second[p[0]]))

    joint_days = {}
    joint_days2 = {}

    for t in range(0, len(digits)):
        days_in_text = day_first_values[t] + " " + day_second_values[t]
        joint_days[digits[t]] = days_in_text 

    for t in range(0, len(digits2)):
        days_in_text2 = day_first_values2[t]
        joint_days2[digits2[t]] = days_in_text2 

    for _ in range(0, len(digits)):
        for t in answer.split():
            if(t.isdigit() and int(t) <= 31 and int(t) >= 10):
                answer = answer.replace(t, joint_days[int(t)])

    for _ in range(0, len(digits2)):
        for t in answer.split():
            if(t.isdigit() and int(t) <= 9):
                answer = answer.replace(t, joint_days2[int(t)])

    return answer

def year_filter(answer):
    answer = drop_characters(answer)

    digits = [int(s) for s in answer.split() if s.isdigit() and int(s) >= 1000]
    
    digits_in_text = []
    for k in digits:
        seperate_year_digits = [int(b) for b in str(k)]
        digits_in_text.append(seperate_year_digits)

    year_first_values = []
    year_second_values = []
    year_third_values = []
    year_fourth_values = []


    for p in digits_in_text:
        year_first_values.append(str(p[0]).replace(str(p[0]), year_first[p[0]]))
        year_second_values.append(str(p[1]).replace(str(p[1]), year_second[p[1]]))
        year_third_values.append(str(p[2]).replace(str(p[2]), year_third[p[2]]))
        year_fourth_values.append(str(p[3]).replace(str(p[3]), year_fourth[p[3]]))

    joint_years = {}
    for t in range(0, len(digits)):
        years_in_text = year_first_values[t] + " " + year_second_values[t] + " " + year_third_values[t] + " " + year_fourth_values[t]
        joint_years[digits[t]] = years_in_text 


    for _ in range(0, len(digits)):
        for t in answer.split():
            if(t.isdigit() and int(t) >= 1000):
                answer = answer.replace(t, joint_years[int(t)])

    return answer

def robertaqna(question, text):
    inputs = roberta_qna_tokenizer(question, text, return_tensors="pt")
    with torch.no_grad():
        outputs = roberta_qna_model(**inputs)

    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()

    predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    answer = roberta_qna_tokenizer.decode(predict_answer_tokens)
            
    if answer == "<s>":
        answer = "javob topilmadi"
    return answer

def english_to_uzbek(text):
    params = '&from=en&to=uz'
    constructed_url = endpoint + path + params

    headers = {
        'Ocp-Apim-Subscription-Key': resource_key,
        'Ocp-Apim-Subscription-Region': region,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }

    body = [{
        'text' : str(text)
    }]
    request = requests.post(constructed_url, headers=headers, json=body)
    response = request.json()
    ans = json.loads(json.dumps(response, sort_keys=True, indent=4, separators=(',', ': ')))


    return ans[0]['translations'][0]['text']

def uzbek_to_english(text):
    params = '&from=uz&to=en'
    constructed_url = endpoint + path + params

    headers = {
        'Ocp-Apim-Subscription-Key': resource_key,
        'Ocp-Apim-Subscription-Region': region,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }

    body = [{
        'text' : str(text)
    }]
    request = requests.post(constructed_url, headers=headers, json=body)
    response = request.json()
    ans = json.loads(json.dumps(response, sort_keys=True, indent=4, separators=(',', ': ')))


    return ans[0]['translations'][0]['text'] 

def drop_word(text, word):
    words = text.split()
    for w in words:
        if w == word:
            words.remove(w)
    answer = ' '.join(words)
    return answer

def drop_characters(text):
    a = text
    b = "!@#$()-_=+%&.,\" "
    
    for char in b:
        a = a.replace(char, " ")    

    return a

stemmer = PorterStemmer()

def tokenize(sentence):
	return nltk.word_tokenize(sentence)

def stem(word):
	return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
	tokenized_sentence = [stem(w) for w in tokenized_sentence]

	bag = np.zeros(len(all_words), dtype=np.float32)
	for idx, w in enumerate(all_words):
		if w in tokenized_sentence:
			bag[idx] = 1.0
	return bag

nltk.download('punkt')

class NeuralNet(nn.Module):
	def __init__(self, input_size, hidden_size, num_classes):
		super(NeuralNet, self).__init__()
		self.l1 = nn.Linear(input_size, hidden_size)
		self.l2 = nn.Linear(hidden_size, hidden_size)
		self.l3 = nn.Linear(hidden_size, num_classes)
		self.relu = nn.ReLU()

	def forward(self, x):
		out = self.l1(x)
		out = self.relu(out)
		out = self.l2(out)
		out = self.relu(out)
		out = self.l3(out)

		#no activation no softmax
		return out
	
def train_model():
	with open('assets/intents.json') as f:
		intents = json.load(f)

	all_words = []
	tags = []
	xy = []

	for intent in intents['intents']:
		tag = intent['tag']
		tags.append(tag)
		for pattern in intent['patterns']:
			w = tokenize(pattern)
			all_words.extend(w)
			xy.append((w, tag))

	ignore_words = ['?', '!', ',', '.']
	all_words = [stem(w) for w in all_words if w not in ignore_words]
	all_words = sorted(set(all_words))
	tags = sorted(set(tags))

	x_train = []
	y_train = []

	for (pattern_sentence, tag) in xy:
		bag = bag_of_words(pattern_sentence, all_words)
		x_train.append(bag)

		label = tags.index(tag)
		y_train.append(label)

	x_train = np.array(x_train)
	y_train = np.array(y_train)


	class ChatDataset(Dataset):
		def __init__(self):
			self.n_samples = len(x_train)
			self.x_data = x_train
			self.y_data = y_train

		def __getitem__(self, index):
			return self.x_data[index], self.y_data[index]

		def __len__(self):
			return self.n_samples


	batch_size = 8
	hidden_size = 8
	output_size = len(tags)
	input_size = len(x_train[0])
	learning_rate = 0.001
	num_epochs = 1000

	dataset = ChatDataset()
	train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	model = NeuralNet(input_size, hidden_size, output_size).to(device)

	# Loss and optimizer
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	# Train the model
	for epoch in range(num_epochs):
		for (words, labels) in train_loader:
			words = words.to(device)
			labels = labels.to(dtype=torch.long).to(device)
			
			# Forward pass
			outputs = model(words)
			# if y would be one-hot, we must apply
			# labels = torch.max(labels, 1)[1]
			loss = criterion(outputs, labels)
			
			# Backward and optimize
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
		if (epoch+1) % 100 == 0:
			print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


	print(f'final loss: {loss.item():.4f}')

	data = {
	"model_state": model.state_dict(),
	"input_size": input_size,
	"hidden_size": hidden_size,
	"output_size": output_size,
	"all_words": all_words,
	"tags": tags
	}

	#mydate = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')

	os.remove("model/data.pth")
	FILE = f"model/data.pth"

	torch.save(data, FILE)

	print(f'training complete. file saved to {FILE}')
	
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INTENTS_FILE = os.path.join(BASE_DIR, 'assets/intents.json')
SIMPLE_QA_MODEL = os.path.join(BASE_DIR, 'model/data.pth')

with open(INTENTS_FILE, 'r') as f:
    intents = json.load(f)

data = torch.load(SIMPLE_QA_MODEL)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]


model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = robot_name

def get_response(msg):

    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                answer = random.choice(intent['responses'])
                return answer
    else:
        return "tushunmadim"

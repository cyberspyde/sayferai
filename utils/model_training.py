stemmer = PorterStemmer()
BASE_DIR = os.path.dirname(os.path.abspath(os.path.join(__file__, '..')))
INTENTS_FILE = os.path.join(BASE_DIR, 'assets/intents.json')

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
	with open(INTENTS_FILE) as f:
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

	SIMPLE_QA_MODEL = os.path.join(BASE_DIR, 'model/data.pth')
	os.remove(SIMPLE_QA_MODEL)
	torch.save(data, SIMPLE_QA_MODEL)

	print(f'training complete. file saved to {SIMPLE_QA_MODEL}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

from my_initializer import synthesize_once, recognize_once, day_filter, year_filter, Logging, robot_name, drop_word, \
        get_current_time, wikipedia, get_response, subjective_knowledge, askQuranInUzbek, json
from get_number_from_text import detect_number

def run_with_no_voice_activation():
    query = recognize_once().lower()
    answer = day_filter(year_filter(get_response(query)))
    print(query)
    #Logging(str(query))
    #Wikipedia knowledge base inclusion (Wikipedia dan qidirish)
    if "haqida" in query and robot_name in query:
        try:
            wiki_question = drop_word(query, robot_name).split(' haqida', 1)[0]
            suggested_wiki_answer = wikipedia.suggest(f"{wiki_question}")             

            # if (suggested_wiki_answer is not None):
            #     wiki_answer = random.choice(suggested_wiki_answer.options)
            try:
                synthesize_once(f"{wiki_question} haqida qidiryabman")
                wiki_answer = wikipedia.summary(wiki_question, sentences=3)
            except wikipedia.DisambiguationError as e:
                s = e.options[-1]
                wiki_answer = wikipedia.summary(s, sentences=3)

            synthesize_once(day_filter(year_filter(wiki_answer)))
        except Exception as e:
            print(e)


    #Soatni so'rash
    if answer == "soatni aytaman":
        currentTime = get_current_time()
        synthesize_once(currentTime)
    

    #Qur'an
    if answer == "Qur'on ilmi":
        synthesize_once("Qur'ondan xohlagan sura va oyatlaringizni so'rashingiz mumkin, bu rejimdan chiqish uchun yakunlash yoki tugatish so'zlaridan foydalaning. ")
        while True:
            request = recognize_once().lower()
            print("query : ", request)
            modified_request = request.replace("’", "'")
            request = detect_number(modified_request)
            print("prediction", request)
            if request == "tugatish" or request == "yakunlash":
                break
            askQuranInUzbek(request)

    #TakeNotes malumotlarni saqlab qolish
    if "eslab qol" in query and robot_name in query.split():
        words = ['eslab', 'qol', 'sayfer']

        for word in words:
            query = drop_word(query, word)

        subjective_knowledge['all'] += ". " + query        
        file = open('assets\\subjective-knowledge.json', 'w')
        file.write(json.dumps(subjective_knowledge))
        file.close()
        synthesize_once("Eslab qoldim")
        print(query)
        
    if answer == "tushunmadim":
        synthesize_once("Tushunmadim")
    
    return query

while True:
    query = run_with_no_voice_activation()
    if query == 'tugatish':
        break
# if "eslab qol" in query and robot_name in query.split():
#     note_first = query.split("eslab qol")[0]
#     if len(note_first) > 10:
#         if "sayfer" in note_first:
#             note = note_first.split("sayfer")[1]
#         else:
#             note = note_first
            
#         if len(note) > 6:
#             subjective_knowledge['all'] += ". " + note        
#             file = open('assets\\subjective-knowledge.json', 'w')
#             file.write(json.dumps(subjective_knowledge))
#             file.close()
#             speech_synthesizer.speak_text_async("Eslab qoldim")
#         else:
#             if "sayfer" in note_first:
#                 note = note_first.split("sayfer")[0]
            
#             if len(note) > 6:
#                 subjective_knowledge['all'] += ". " + note        
#                 file = open('assets\\subjective-knowledge.json', 'w')
#                 file.write(json.dumps(subjective_knowledge))
#                 file.close()
#                 speech_synthesizer.speak_text_async("Eslab qoldim")
#             else:
#                 print("Given information isn't valid first")
#     else:
#         note_second = query.split("eslab qol")[1]
#         if len(note_second) > 10:
#             if "sayfer" in note_second:
#                 note = note_second.split("sayfer")[0]
#             else:
#                 note = note_second
                
#             if len(note) > 6:
#                 subjective_knowledge['all'] += ". " + note        
#                 file = open('assets\\subjective-knowledge.json', 'w')
#                 file.write(json.dumps(subjective_knowledge))
#                 file.close()
#                 speech_synthesizer.speak_text_async("Eslab qoldim")
#             else:
#                 print("Given information isn't valid")
#         else:
#             if "sayfer" in note_second:
#                 note = note_second.split("sayfer")[1]
#             if len(note) > 6:
#                 subjective_knowledge['all'] += ". " + note        
#                 file = open('assets\\subjective-knowledge.json', 'w')
#                 file.write(json.dumps(subjective_knowledge))
#                 file.close()
#                 speech_synthesizer.speak_text_async("Eslab qoldim")
#             else:
#                 print("Given information isn't valid second")
    

     
    #speech_synthesizer.speak_text_async("Ilmiy ma'lumotlar bazasi topilmadi")
from utils.my_initializer import synthesize_once, recognize_once, day_filter, year_filter, Logging, robot_name, drop_word, \
        get_current_time, wikipedia, get_response, subjective_knowledge, askQuranInUzbek, json, text2num

def run_with_no_voice_activation():
    try:
        #query = recognize_once().lower()
        query = recognize_once().lower()
        answer = day_filter(year_filter(get_response(query)))
        print(query)
        status = True
        #Wikipedia 
        if "haqida" in query and robot_name in query:
            try:
                wiki_question = drop_word(query, robot_name).split(' haqida', 1)[0]
                suggested_wiki_answer = wikipedia.suggest(f"{wiki_question}")             
                try:
                    synthesize_once(f"{wiki_question} haqida qidiryabman")
                    wiki_answer = wikipedia.summary(wiki_question, sentences=3)
                except wikipedia.DisambiguationError as e:
                    s = e.options[-1]
                    wiki_answer = wikipedia.summary(s, sentences=3)

                synthesize_once(day_filter(year_filter(wiki_answer)))
                status = False
            except Exception as e:
                print(e)
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
            if request == "tugatish" or request == "yakunlash":
                break
            print("query : ", request)
            modified_request = request.replace("’", "'")
            request = text2num(modified_request)
            print("prediction", request)
            askQuranInUzbek(request)

    #TakeNotes malumotlarni saqlab qolish
    if "eslab qol" in query and robot_name in query.split():
        words = ['eslab', 'qol', robot_name]

        for word in words:
            query = drop_word(query, word)

        subjective_knowledge['all'] += ". " + query        
        file = open('assets\\subjective-knowledge.json', 'w')
        file.write(json.dumps(subjective_knowledge))
        file.close()
        synthesize_once("Eslab qoldim")
        print(query)
        status = False
        

    if answer == "tushunmadim" and status == True:
        synthesize_once("Tushunmadim")
    
    return query

while True:
    query = run_with_no_voice_activation()
    if query == 'tugatish':
        break
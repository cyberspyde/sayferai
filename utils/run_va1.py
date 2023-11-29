from utils.my_initializer import synthesize_once, recognize_once, day_filter, year_filter, Logging, robot_name, drop_word, \
        get_current_time, wikipedia, get_response, subjective_knowledge, askQuranInUzbek, json, text2num
import playsound

def run_with_voice_activation():
    sayfer_status = "offline" 

    if(sayfer_status == "offline"):
        query = recognize_once().lower()
        print(query)
        if(robot_name in query.lower()):
            sayfer_status = "online"

    if(sayfer_status == "online"):

        playsound('assets\\audio.mp3', block=False)
        query = recognize_once().lower()
        answer = day_filter(year_filter(get_response(query)))
        print(query)
        print(answer)

        #Wikipedia
        if ("haqida" in query):
            sayfer_status = "offline"
            try:
                wiki_question = query.split(' haqida', 1)[0]
                suggested_wiki_answer = wikipedia.suggest(f"{wiki_question}")             

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
        sayfer_status = "online"
        if answer == "soatni aytaman":
            currentTime = get_current_time()
            synthesize_once(currentTime)
            sayfer_status = "offline"


        if(sayfer_status != "offline"):
            #Buyruq aniqlanmagan holatda
            if(answer == "tushunmadim"):
                sayfer_status = "offline"
                synthesize_once("tushunmadim")
            else:
                answer = day_filter(year_filter(answer))
                synthesize_once(answer)
def run_with_voice_activation():
    sayfer_status = "offline" 

    if(sayfer_status == "offline"):
        query = speech_recognizer.recognize_once_async().get().text
        print(query)
        if(robot_name in query.lower()):
            sayfer_status = "online"

    if(sayfer_status == "online"):

        playsound('assets\\audio.mp3', block=False)
        query = speech_recognizer.recognize_once_async().get().text
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
                    speech_synthesizer.speak_text_async(f"{wiki_question} haqida qidiryabman").get()
                    wiki_answer = wikipedia.summary(wiki_question, sentences=3)
                except wikipedia.DisambiguationError as e:
                    s = e.options[-1]
                    wiki_answer = wikipedia.summary(s, sentences=3)

                speech_synthesizer.speak_text_async(day_filter(year_filter(wiki_answer))).get()

            except Exception as e:
                print(e)

        #Soatni so'rash
        sayfer_status = "online"
        if answer == "soatni aytaman":
            currentTime = get_current_time()
            speech_synthesizer.speak_text_async(currentTime).get()
            sayfer_status = "offline"


        if(sayfer_status != "offline"):
            #Buyruq aniqlanmagan holatda
            if(answer == "tushunmadim"):
                sayfer_status = "offline"
                speech_synthesizer.speak_text_async("tushunmadim").get()
            else:
                answer = day_filter(year_filter(answer))
                speech_synthesizer.speak_text_async(answer).get()
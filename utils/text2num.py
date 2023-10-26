import argparse

parser = argparse.ArgumentParser()
parser.add_argument('text', nargs="?", help="Raqam yoki sonlarni yozma holatda kiriting")
args = parser.parse_args()

keywords = {
    "bir": "1",
    "ikki": "2",
    "uch": "3",
    "to'rt": "4",
    "besh": "5",
    "olti": "6",
    "yetti": "7",
    "sakkiz": "8",
    "to'qqiz": "9",
    "o'n": "10",
    "yigirma": "20",
    "o'ttiz": "30",
    "qirq": "40",
    "ellik": "50",
    "oltmish": "60",
    "yetmish": "70",
    "sakson": "80",
    "to'qson": "90",
    "yuz": "00",
    "ming": "000",
    "million": "000000",
    "milliard": "000000000"
}

exact_queries = ["yuz", "ming", "million", "milliard"]
exact_keys = [100, 1000, 1000000, 1000000000]
onlik = ["0", "00", "000", "000000", "000000000"]

def text2num(query):
    try:
        if query is None:
            raise Exception("Yozuvli raqam kiritilmadi!, Skript nomidan keyin joy qoldirib, raqamni yozuv ko'rinishida apostrof ichida kiriting!\n\
                            Masalan 'python text2num.py `bir yuz yigirma olti`'")
        million_exist = False
        yuzlik_exist = False
        minglik_exist = False
        modified = query.split(" ")
        for k in range(0, len(modified)):
            if modified[k] == "million":
                million_exist = True
            if modified[k] == "ming":
                minglik_exist = True
            if modified[k] == "yuz":
                yuzlik_exist = True
            modified[k] = keywords[modified[k]]
        try:
            for t in range(0, len(modified)):
                if modified[t] in onlik:
                    modified[t-1] += modified[t]
                    modified.remove(modified[t])
        except IndexError:
            pass

        if minglik_exist and not million_exist:
            for t in range(0, len(modified)):
                if len(modified[t]) < 4:
                    modified[t] += "000"
                if len(modified[t+1]) >= 4 and len(modified[t]) >= 4:
                    break
                else:
                    modified[t+1] += "000"
                    if len(modified[t+2]) == 4:
                        break
                    else:
                        modified[t+2] += "000"
                        break

        if million_exist:
            try:
                for p in range(0, len(modified)):
                    if len(modified[p]) < 7:
                        modified[p] += "000000"
                    if len(modified[p]) == 7:
                        if len(modified[p+1]) < 4:
                            modified[p+1] += "000"
                            if len(modified[p+2]) < 4:
                                modified[p+2] += "000"
                        break
            except IndexError:
                pass

        result = 0
        for m in modified:
            result += int(m)

        if query in exact_queries:
            result = exact_keys[exact_queries.index(query)]
        
        #print("Natija : ", result)
        return result
    except IndexError as e:
        print("Indeksda xatolik yuz berdi. ", e)
    except Exception as argErr:
        print(argErr)
    except KeyError as keyErr:
        print("Bunday raqam mavjud emas ", keyErr)


#print(text2num("bir yuzi o'n"))
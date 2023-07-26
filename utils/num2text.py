import argparse

parser = argparse.ArgumentParser()
parser.add_argument('text', nargs="?", help="Raqam yoki sonlarni son holatda kiriting")
args = parser.parse_args()

def num2text(number):
    try:
        number = int(number)
        # uzbek raqamlari
        words = {
            "0": "nol",
            "1": "bir",
            "2": "ikki",
            "3": "uch",
            "4": "to'rt",
            "5": "besh",
            "6": "olti",
            "7": "yetti",
            "8": "sakkiz",
            "9": "to'qqiz",
            "10": "o'n",
            "20": "yigirma",
            "30": "o'ttiz",
            "40": "qirq",
            "50": "ellik",
            "60": "oltmish",
            "70": "yetmish",
            "80": "sakson",
            "90": "to'qson",
            "100": "yuz",
            "1000": "ming",
            "1000000": "million",
            "1000000000": "milliard"
        }

        # ikki xonali raqamni textga o'girish
        def convert_two_digits(num):
            if num < 10:
                return words[str(num)]
            elif num % 10 == 0:
                return words[str(num)]
            else:
                return words[str(num // 10 * 10)] + " " + words[str(num % 10)]

        # 3 xonali raqamni textga o'girish
        def convert_three_digits(num):
            hundreds = num // 100
            remainder = num % 100
            if remainder == 0:
                return words[str(hundreds)] + " yuz"
            else:
                return words[str(hundreds)] + " yuz " + convert_two_digits(remainder)

        # berilgan raqamni textga o'girish
        if number < 0:
            return "minus " + num_to_text_uzbek(abs(number))
        elif number < 100:
            return convert_two_digits(number)
        elif number < 1000:
            return convert_three_digits(number)
        elif number < 1000000:
            thousands = number // 1000
            remainder = number % 1000
            if remainder == 0:
                return num_to_text_uzbek(thousands) + " ming"
            else:
                return num_to_text_uzbek(thousands) + " ming " + num_to_text_uzbek(remainder)
        elif number < 1000000000:
            millions = number // 1000000
            remainder = number % 1000000
            if remainder == 0:
                return num_to_text_uzbek(millions) + " million"
            else:
                return num_to_text_uzbek(millions) + " million " + num_to_text_uzbek(remainder)
        elif number < 1000000000000:
            billions = number // 1000000000
            remainder = number % 1000000000
            if remainder == 0:
                return num_to_text_uzbek(billions) + " milliard"
            else:
                return num_to_text_uzbek(billions) + " milliard " + num_to_text_uzbek(remainder)

        return "Xatolik"
    except Exception as e:
        print(e)
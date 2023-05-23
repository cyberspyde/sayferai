from utils.run_va1 import run_with_voice_activation
from utils.run_va0 import run_with_no_voice_activation
from utils.my_initializer import voice_activation

if __name__ == "__main__":
    while True:
        if voice_activation == "False":
            run_with_no_voice_activation()
        elif voice_activation == "True":
            run_with_voice_activation()
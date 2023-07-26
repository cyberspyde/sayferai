set /a Index=0

setlocal enabledelayedexpansion

for /r %%i in (Quran\\*.mp3) do ( 
    rename "%%i" "Quran\\!Index!.mp3"
    set /a Index+=1
)
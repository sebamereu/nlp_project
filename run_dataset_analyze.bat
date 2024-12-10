@echo off
REM Definisci il percorso dell'eseguibile Python
set PYTHON_EXEC=C:\Users\sebam\anaconda3\envs\nlpvenv\python.exe

REM Definisci il percorso dello script Python
set SCRIPT_PATH=C:\Users\sebam\Desktop\NLP\progettoMereuNLP\pan-clef-2024-oppositional-main\pan-clef-2024-oppositional-main\dataset_analyze.py

REM Definisci il percorso del file JSON
set FILE_PATH=C:\Users\sebam\Desktop\NLP\progettoMereuNLP\pan-clef-2024-oppositional-main\pan-clef-2024-oppositional-main\dataset\dataset_en_test.json

REM (Facoltativo) Definisci la directory di output per i grafici
set PLOT_DIR=C:\Users\sebam\Desktop\plots

REM Esegui il comando
"%PYTHON_EXEC%" "%SCRIPT_PATH%" --file_path "%FILE_PATH%" --plot_dir "%PLOT_DIR%"

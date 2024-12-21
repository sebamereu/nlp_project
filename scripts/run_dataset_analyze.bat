@echo off
REM Definisci il percorso dell'eseguibile Python
set PYTHON_EXEC=C:\Users\sebam\anaconda3\envs\nlpvenv\python.exe

REM Definisci il percorso dello script Python
set SCRIPT_PATH=C:\Users\sebam\Desktop\NLP\progettoMereuNLP\pan-clef-2024-oppositional-main\pan-clef-2024-oppositional-main\dataset_analyze.py

REM Analizza il dataset di test
set FILE_PATH_TEST=C:\Users\sebam\Desktop\NLP\progettoMereuNLP\pan-clef-2024-oppositional-main\pan-clef-2024-oppositional-main\dataset\dataset_en_test.json
set PLOT_DIR_TEST=C:\Users\sebam\Desktop\NLP\progettoMereuNLP\pan-clef-2024-oppositional-main\pan-clef-2024-oppositional-main\plot\test
"%PYTHON_EXEC%" "%SCRIPT_PATH%" --file_path "%FILE_PATH_TEST%" --plot_dir "%PLOT_DIR_TEST%"

REM Analizza il dataset di training
set FILE_PATH_TRAIN=C:\Users\sebam\Desktop\NLP\progettoMereuNLP\pan-clef-2024-oppositional-main\pan-clef-2024-oppositional-main\dataset\dataset_en_train.json
set PLOT_DIR_TRAIN=C:\Users\sebam\Desktop\NLP\progettoMereuNLP\pan-clef-2024-oppositional-main\pan-clef-2024-oppositional-main\plot\train
"%PYTHON_EXEC%" "%SCRIPT_PATH%" --file_path "%FILE_PATH_TRAIN%" --plot_dir "%PLOT_DIR_TRAIN%"

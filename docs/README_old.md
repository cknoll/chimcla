 Aktuelles Vorgehen:


## Stage 1:

- 20 MB png → 0.2 MB jpg
    - `py3 stage_0f_resize_and_jpg.py /home/ck/mnt/XAI-DIA-gl/Carsten/bilder_roh_aus_peine_ab_2023-07-31`
    - alt: `mogrify -monitor -format jpg -resize 1000 -path ../bilder_jpg2 *.png`
- Bilder ohne Formen erkennen
    - `py3 stage_1a_empty_slot_detection.py /home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2`
- Bilder croppen
    - `py3 stage_1b_cropping.py /home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2`
- Chunks erstellen
    - `py3 stage_1b2_chunk_splitter.py /home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2/cropped`
- Bilder aufhellen
    - `py3 stage_1c_shading_correction.py /home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2/cropped/chunk001 &`
    - ca 20s für 1000 Bilder
- grobe Klassifikation manuell -> csv
    - image_classification/PyQt-image-annotation-tool/main.py
- csv -> Verzeichnisse
    - `py3 stage1f_evaluate_csv__copy_to_dir.py cropped/chunk0XYZ_shading_corrected/output/assigned_classes.csv`



### Probleme:

2023-06-26_07-54-48_C0.jpg could not find bbox, even with detrend and different threshold



## Stage 2 (wip):

- für jedes Bild: für jedes Stäbchen ein Histogramm:
    - `py3 stage_2b_bar_C0_hist_dict.py --img_dir /home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2a/cropped/chunk000_stage1_completed/C0`
    - Aus Zeitgründen erstmal nur 25 Bilder
- Für jede Stäbchenposition: Histogramme aggregieren:
    - http://localhost:8888/notebooks/XAI-DIA/image_classification/stage2/b_03_histogram-evaluation.ipynb -> "_total_res.dill"
- Ausreißer finden
    - http://localhost:8888/notebooks/XAI-DIA/image_classification/stage2/b_04_find_annomalies.ipynb
    - bzw.: `py3 stage_2c_hist_based_anomaly_detection.py`
- Dort manuell die falsch positiven in separates Verzeichnis sortieren
- Histogramme (minimalinvasiv) so korrigieren, dass falsch positive akzeptiert werden
    - http://localhost:8888/notebooks/XAI-DIA/image_classification/stage2/b_05_adapt_histograms.ipynb
    - → "_total_res.dill" wird angepasst


## Stage 3

- Klassifikation komplettes VZ:
    - `py3 stage_3a_hist_based_classification.py --img_dir /home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2a/cropped/chunk003_shading_corrected/ --suffix _chunk003_complete`
- Flächen-basierte Klassifikation (wip):
    - `py3 stage_3b_hist_area_based_classification.py --img /home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2a/cropped/chunk000_shading_corrected/2023-06-26_06-47-13_C50.jpg`
    - `py3 stage_3b_hist_area_based_classification.py --img_dir /home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2a/cropped/chunk000_shading_corrected/ --suffix _chunk000_complete --limit 20`


---

# geplantes weiteres Vorgehen


- profiling und mehrfachausführung ausschließen.

- falsch positive handhaben -> Histogramme anpassen
- `dicts/_total_res.dill`


- Riegel separieren -> statistische Verfahren zum Labeling verwenden


-> Ziel 02.10.: 1000 Bilder vorsortieren


- feinere Klassifikation
- 1. Testtraining


---

### Romys neue Experimentaldaten:

- `py3 stage_3d_create_experimental_data.py --img /home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2a/cropped/chunk001_shading_corrected/2023-06-26_08-47-39_C50.jpg --suffix _psy1`
- `py3 stage_3d_create_experimental_data.py --img_dir /home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2a/cropped/chunk001_shading_corrected --suffix _psy01 -bm 0 -bv 50`
- `py3 stage_3e_create_experimental_data_csv.py experimental_imgs_psy01_bm0_bv50`
- zum interaktiven Experimentieren: `pytest -sk 00_ips tests/`




```
py3 stage_3d_create_experimental_data.py --img /home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2a/cropped/chunk001_shading_corrected/2023-06-26_08-51-39_C50.jpg --suffix _psy1 -std -bv 50
py3 stage_3d_create_experimental_data.py --img /home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2a/cropped/chunk001_shading_corrected/2023-06-26_08-51-39_C50.jpg --suffix _psy1 -std -bv 70
py3 stage_3d_create_experimental_data.py --img /home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2a/cropped/chunk001_shading_corrected/2023-06-26_08-51-39_C50.jpg --suffix _psy1 -std -bv 90
py3 stage_3d_create_experimental_data.py --img /home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2a/cropped/chunk001_shading_corrected/2023-06-26_08-51-39_C50.jpg --suffix _psy1 -std -bv 110
py3 stage_3d_create_experimental_data.py --img /home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2a/cropped/chunk001_shading_corrected/2023-06-26_08-51-39_C50.jpg --suffix _psy1 -std -bv 130
py3 stage_3d_create_experimental_data.py --img /home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2a/cropped/chunk001_shading_corrected/2023-06-26_08-51-39_C50.jpg --suffix _psy1 -std -bv 150
py3 stage_3d_create_experimental_data.py --img /home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2a/cropped/chunk001_shading_corrected/2023-06-26_08-51-39_C50.jpg --suffix _psy1 -std -bv 170
py3 stage_3d_create_experimental_data.py --img /home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2a/cropped/chunk001_shading_corrected/2023-06-26_08-51-39_C50.jpg --suffix _psy1 -std -bv 190
py3 stage_3d_create_experimental_data.py --img /home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2a/cropped/chunk001_shading_corrected/2023-06-26_08-51-39_C50.jpg --suffix _psy1 -std -bv 210




# img_dir
py3 stage_3d_create_experimental_data.py --img_dir /home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2a/cropped/chunk001_shading_corrected --suffix _psy01 -bm 0 -bv 50
py3 stage_3d_create_experimental_data.py --img_dir /home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2a/cropped/chunk001_shading_corrected --suffix _psy01 -bm 0 -bv 70
py3 stage_3d_create_experimental_data.py --img_dir /home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2a/cropped/chunk001_shading_corrected --suffix _psy01 -bm 0 -bv 90
py3 stage_3d_create_experimental_data.py --img_dir /home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2a/cropped/chunk001_shading_corrected --suffix _psy01 -bm 0 -bv 110
py3 stage_3d_create_experimental_data.py --img_dir /home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2a/cropped/chunk001_shading_corrected --suffix _psy01 -bm 0 -bv 130
py3 stage_3d_create_experimental_data.py --img_dir /home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2a/cropped/chunk001_shading_corrected --suffix _psy01 -bm 0 -bv 150
py3 stage_3d_create_experimental_data.py --img_dir /home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2a/cropped/chunk001_shading_corrected --suffix _psy01 -bm 0 -bv 170
py3 stage_3d_create_experimental_data.py --img_dir /home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2a/cropped/chunk001_shading_corrected --suffix _psy01 -bm 0 -bv 190



Sonntag:

py3 stage_3d_create_experimental_data.py --img_dir /home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2a/cropped/chunk007_shading_corrected --suffix _psy01 -bm 0 -bv 60




# problematische Bilder beim Erzeugen der CSV-Datei (KeyError):

'2023-06-27_04-19-04_C0'
'2023-06-27_09-38-31_C5'


```

### History evaluation


This command does what we want:
```
chimcla_ced --img_dir /home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2a/cropped/chunk001_shading_corrected/ --suffix _history_test_x -H --limit 1 -np

```

next step: add step-history from log lines to the same image

`chimcla_step_history_eval -l ~/mnt/XAI-DIA-gl/Carsten/logs/classifier-2023-07-10_since_2023-06-26.log --csv-mode stage3_results__history_test_y\* 300`



#### für Versuchspersonen:

jeweils drei Bilder:
- unverfälschte Form
- Form wo die kritischen Pixel der kritischen Riegel eingefärbt wurden (unabhängig von der Helligkeit) + Text: Anzahl der auffälliger Riegel: ...
- Form wo die kritischen Pixel der kritischen Riegel eingefärbt wurden (abhängig von der Helligkeit) + Text: Anzahl der auffälliger Riegel: ...


#### Für Romy:

Wie bisher, aber die original-Bilddaten für die Anzeige verwenden (nicht für die Verarbeitung)



Ergebnisse Besprechung (2023-11-27 10:55):


- Heatmap muss nicht über existierendes Bild drüber gelegt werden, sondern kann alleine stehen. Farbverlauf (Schwarz->Rot), Jet, ... (Vorschläge machen)
- [x] uniformes Einfärben einzelner Stäbchen mit "mittlerem Wert"
    - parametrisieren für einfacheres Erstellen von Vorschlägen
- Tabelle (csv): Formen-Bild, Anzahl betroffener Riegel, Summe kritischer Pixel, max 95% quantil, Anzahl der Pixel, die >= q95 sind
- Methodische Eckpunkte dokumentieren

- [] Randbereiche bei kritischen Pixeln möglichst nicht einfärben
- [x] Woher kommt der Blaustich? -> BGR statt RGB


--


- [ ] je 100 Bilder mit verschiedenen Farbeinstellungen
    - im suffix notieren: farbeinstellungen
    - im Dateinamen notieren gesamt-Kritikalität


--

- [] Hard Blend 150 oder Median der Softblend-Werte
- [] Varianz in Tabelle und annotieren
- [] Tab separiert
- S123_2023-06-26_08-51-33_C50_exp_soft_50 debuggen

--

- S123_2023-06-26_08-51-33_C50_exp_soft_50 debuggen

py3 stage_3d_create_experimental_data.py --img /home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2a/cropped/chunk001_shading_corrected/2023-06-26_08-51-33_C50.jpg --suffix _psy1 -std -bv 210

-> Es kommt nur S94 raus
Zelle B19 wird hervorgehoben

Was bedeuten die Zahlen? mittelwert und Standardabweichung


b17 Analyse zum Debuggen:

py3 stage_3b_hist_area_based_classification.py --img /home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2a/cropped/chunk001_shading_corrected/2023-06-26_08-51-33_C50.jpg

---

py3 stage_3e_create_experimental_data_csv.py --diagram "/home/ck/mnt/XAI-DIA-gl/Carsten/_selection_round3_2024-01-16-experimental-bilder-hard110-soft60"



--

Statistisches Übersichtsdiagram Fehlerfläche vs. Helligkeit

/home/ck/mnt/XAI-DIA-gl/Carsten/_selection_round3_2024-01-16-experimental-bilder-hard110-soft60/definitely keep


# CNN-Trainingsdaten

Befehl 2024-01-29:
py3 stage_3b_hist_area_based_classification.py -ad --img_dir /home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2a/cropped/chunk000_shading_corrected/ --suffix _chunk000_complete2 --limit 8

dann aux_rename_cell_imgs.py


2024-02-23 11:47:00

py3 stage_3b_hist_area_based_classification.py -ad --img_dir /home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2a/cropped/chunk001_shading_corrected/ --suffix _chunk001_cell_training2

-> alle Bilder >= S60 nach /home/ck/XAI-DIA-bilder kopiert


-> py3 aux_rename_cell_imgs.py /home/ck/XAI-DIA-bilder

-> PyQt-image-annotation-tool anwenden

->


## Übersicht über bisher schon gelabelte Bilder

- Januar 2024 (Carsten): /home/ck/XAI-DIA-bilder/___chunk_0000 (1329)
- Februar 2024 (Carsten): /home/ck/XAI-DIA-bilder/__chunk_0001 (355) (manuell ab s60 ausgewählt)
-      Ursprung: /home/ck/iee-ge/XAI-DIA/image_classification/stage2/critical_hist_chunk001_cell_training2 (916 Dateien ab s20)
- April 2024
    - chocolate_rename_cell_imgs .


nächste Schritte: Hochlanden, testhalber labeln, runterladen


### debugging:
2023-06-26_08-49-11_C50


stage_3b_hist_area_based_classification.py --img /home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2a/cropped/chunk001_shading_corrected/2023-06-26_08-49-11_C50.jpg --suffix _chunk001_cell_training --adgen-mode


CS_LIMIT = 40 wirkungslos?


# Datensparstrategie:

Stand:

at 15:53:27 ❯ dirsizes
sizes of directories in megabytes
0       2023-05-26
1       bilder_jpg0
1       Doku fuer Peine
1       log-07-43.txt
1       README.md
1       SECAI_Tetzlaff_Montúfar.docx
1       Statistik.jpeg
1       tmp123
2       bilder_jpg_speziell
9       png_files.txt
12      Plots.pptx
22      tmp1
43      tmp2
52      bilder_png-to-jpg-experiments
56      0_besprechungen
75      0_lndw
91      bilder_jpg2a_demo
103     tmp_shared
147     2023-07-17
173     2023-03-16
232     logs
343     PC_backup
345     bilder_jpg3
631     2023-07-18
2821    bilder_jpg
3794    Bilder aus Peine
6579    bilder_jpg2a
6804    bilder_jpg2
55130   bilder_roh_aus_peine_ab_2023-08-29
81928   bilder_roh_aus_peine_ab_2023-09-27
107669  bilder_roh_aus_peine_ab_2023-05-25
118571  bilder_roh_backup
135752  bilder_roh_aus_peine_ab_2023-10-18
204081  bilder_roh_aus_peine_ab_2023-09-04
215061  2023-05-26_Bilder_aus_Peine
247165  bilder_roh_aus_peine_ab_2023-06-05
248164  tmp
298757  bilder_roh_aus_peine_ab_2023-08-30
343305  bilder_roh_aus_peine_ab_2023-06-26
420883  bilder_roh_aus_peine_ab_2023-09-18
439428  bilder_roh_aus_peine_ab_2023-08-30_ab16Uhr
461743  bilder_roh_aus_peine_ab_2023-07-31
3399955 insgesamt




---
















Probleme:
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-30-10_C50.jpg a 2

py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-39-27_C50.jpg b 12

py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-50-10_C50.jpg b 11
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-50-10_C50.jpg b 12
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-50-10_C50.jpg b 13
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-50-10_C50.jpg b 14
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-50-10_C50.jpg b 15
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-50-10_C50.jpg b 16
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-50-10_C50.jpg b 17


---


py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-22-47_C50.jpg c 8
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-23-41_C50.jpg c 8
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-28-54_C50.jpg b 11
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-28-54_C50.jpg b 12
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-28-54_C50.jpg b 13
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-29-54_C50.jpg b 8
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-32-14_C50.jpg c 13
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-32-34_C50.jpg c 8
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-34-48_C50.jpg b 13
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-35-17_C50.jpg a 14
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-35-17_C50.jpg b 9
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-35-17_C50.jpg b 12
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-35-17_C50.jpg b 13
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-35-17_C50.jpg b 14
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-35-17_C50.jpg b 15
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-35-17_C50.jpg b 16
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-35-17_C50.jpg b 17
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-35-17_C50.jpg b 18
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-35-17_C50.jpg b 19
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-35-17_C50.jpg c 14
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-37-18_C50.jpg b 19
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-39-48_C50.jpg b 12
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-39-53_C50.jpg b 12
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-39-53_C50.jpg b 13
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-39-53_C50.jpg b 14
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-39-53_C50.jpg b 15
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-39-53_C50.jpg b 16
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-39-53_C50.jpg b 17
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-39-53_C50.jpg b 18
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-39-53_C50.jpg b 19
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-40-43_C50.jpg c 8
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-41-43_C50.jpg a 11
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-41-43_C50.jpg a 12
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-41-43_C50.jpg a 14
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-41-43_C50.jpg b 8
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-41-43_C50.jpg b 9
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-41-43_C50.jpg b 10
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-41-43_C50.jpg b 11
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-41-43_C50.jpg b 12
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-41-43_C50.jpg b 13
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-41-43_C50.jpg b 14
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-41-43_C50.jpg b 15
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-41-43_C50.jpg b 16
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-41-43_C50.jpg b 17
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-41-43_C50.jpg b 18
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-41-43_C50.jpg b 19
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-41-43_C50.jpg c 14
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-41-50_C50.jpg b 10
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-41-50_C50.jpg b 12
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-41-50_C50.jpg b 13
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-41-50_C50.jpg b 16
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-41-50_C50.jpg b 17
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-44-01_C50.jpg b 12
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-44-01_C50.jpg b 13
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-44-01_C50.jpg b 16
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-44-01_C50.jpg b 17
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-44-01_C50.jpg b 18
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-44-01_C50.jpg b 19
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-44-25_C50.jpg c 8
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-47-13_C50.jpg c 8
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-47-55_C50.jpg a 14
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-47-55_C50.jpg b 11
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-47-55_C50.jpg b 12
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-47-55_C50.jpg b 13
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-47-55_C50.jpg b 14
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-47-55_C50.jpg b 15
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-47-55_C50.jpg b 16
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-47-55_C50.jpg b 17
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-47-55_C50.jpg b 18
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-47-55_C50.jpg b 19
py3 stage_2a_bar_selection.py /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-49-55_C50.jpg c 14







 1   2   3   4  5   6   7   8   9  10  11  12  13  14  15  16 17  18  19  20  21  22 23  24  25  26  27

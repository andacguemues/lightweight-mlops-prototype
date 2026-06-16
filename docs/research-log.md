# Research Log

## 2026-06-16

### Ziel
Initialisierung des Praxisprojekts und Festlegung der ersten technischen Entscheidungen.

### Entscheidungen
- Use Case: UCI Bank Marketing
- Repo-Name: lightweight-mlops-prototype
- Setup: Python 3.12 mit uv
- Fokus: lokal ausführbarer MLOps-Prototyp mit reproduzierbarer Pipeline, Modellversionierung, Prediction-Service und Monitoring

### Begründung
Der UCI Bank Marketing Datensatz eignet sich als tabellarischer Klassifikationsfall für einen leichtgewichtigen MLOps-Prototyp. Der Datensatz ist ausreichend realistisch, aber lokal gut verarbeitbar.

### Umgesetzt
- Projektordner angelegt
- Git initialisiert
- Grundstruktur geplant

### Offene Punkte
- Datensatz herunterladen und dokumentieren
- Baseline-Notebook erstellen
- erste Metriken festlegen

### Nächster Schritt
Aufbau des notebook-zentrierten Baseline-Workflows.

## 2026-06-16

### Ziel
Beginn von Schritt 3: Datensatz beziehen und notebook-zentrierten Baseline-Workflow vorbereiten.

### Entscheidungen
- Der UCI Bank Marketing Datensatz wird als Demonstrationsszenario verwendet.
- Als Datei wird `bank-additional-full.csv` genutzt.
- Der Baseline-Workflow wird bewusst als Notebook umgesetzt, um später einen Vergleich zum MLOps-orientierten Workflow zu ermöglichen.

### Umgesetzt
- Datensatz heruntergeladen und lokal unter `data/raw/` abgelegt.
- Data Card angelegt.
- Baseline-Notebook vorbereitet.
- Erste Modellpipeline mit Preprocessing und Logistic Regression geplant.

### Offene Punkte
- Baseline-Notebook vollständig ausführen.
- Baseline-Metriken dokumentieren.
- Schwächen des Notebook-Workflows festhalten.

### Nächster Schritt
Notebook-Baseline ausführen und Ergebnisse versionieren.
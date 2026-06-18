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

### Technische Beobachtung

Beim Erstellen des Baseline-Notebooks wurde festgestellt, dass Jupyter den Projekt-Hauptordner als aktuellen Arbeitsordner verwendet. Daher mussten die relativen Pfade im Notebook angepasst werden:

- Datenpfad: `data/raw/bank-additional/bank-additional-full.csv`
- Modellpfad: `models/baseline_logistic_regression.joblib`

Diese Beobachtung ist relevant für die spätere Pipeline-Implementierung, da Pfade dort zentral und reproduzierbar definiert werden sollen.

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

## 2026-06-16

### Ziel
Überführung des notebook-zentrierten Baseline-Workflows in eine reproduzierbare Pipeline.

### Entscheidungen
- Die Datenaufbereitung, das Training und die Evaluation werden aus dem Notebook in separate Python-Skripte überführt.
- Zentrale Parameter werden in `params.yaml` dokumentiert.
- DVC wird zur Definition und Reproduktion der Pipeline verwendet.
- MLflow wird noch nicht integriert, sondern als nächster Schritt nach der reproduzierbaren Pipeline eingeplant.

### Umgesetzt
- Download-Skript für den Datensatz erstellt.
- Prepare-Skript für Train/Test-Split erstellt.
- Train-Skript mit Preprocessing und Logistic Regression erstellt.
- Evaluate-Skript mit zentralen Klassifikationsmetriken erstellt.
- DVC initialisiert.
- DVC-Pipeline mit den Stages `download_data`, `prepare_data`, `train_model` und `evaluate_model` definiert.
- Pipeline mit `dvc repro` erfolgreich ausgeführt.

### Ergebnis
Der ML-Workflow kann nun außerhalb des Notebooks reproduzierbar ausgeführt werden. Dadurch entsteht ein erster MLOps-orientierter Workflow, der später gegenüber dem notebook-zentrierten Baseline-Workflow evaluiert werden kann.

### Offene Punkte
- MLflow Tracking und Modellversionierung integrieren.
- Prediction-Service entwickeln.
- Monitoring-Komponente entwickeln.

### Nächster Schritt
Integration von Experiment Tracking und Modellversionierung.

## 2026-06-17

### Ziel

Integration von Experiment Tracking und Modellversionierung in die reproduzierbare DVC-Pipeline.

### Entscheidungen

- MLflow wird für Experiment Tracking und Modellregistrierung verwendet.
- Als lokales Backend wird eine SQLite-Datenbank eingesetzt.
- Modelle werden erst registriert, wenn ein definierter Mindestwert für ROC-AUC erreicht wird.
- Die aktuell registrierte Modellversion erhält zunächst den Alias `candidate`.
- MLflow-Metadaten und Modellartefakte werden nicht über Git versioniert.

### Umgesetzt

- Evaluationsskript um MLflow Tracking erweitert.
- Modellparameter und Evaluationsmetriken werden protokolliert.
- Git-Commit sowie Hashwerte von Testdaten und Modellartefakt werden als Tags gespeichert.
- Modell wird mit Input Example und Model Signature gespeichert.
- Registry-Skript erstellt.
- DVC-Pipeline um die Stage `register_model` erweitert.
- Modellversion erfolgreich in der MLflow Model Registry registriert.

### Ergebnis

Der MLOps-Workflow ermöglicht nun die Zuordnung eines trainierten Modells zu Parametern, Metriken, Code-Stand und Datenartefakten. Akzeptierte Modelle werden versioniert in einer lokalen Model Registry verwaltet.

### Offene Punkte

- Prediction-Service entwickeln.
- Modell anhand des Registry-Alias laden.
- Monitoring-Komponente entwickeln.

### Nächster Schritt

Entwicklung eines Prediction-Service für die registrierte Modellversion.
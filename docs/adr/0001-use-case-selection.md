# ADR-0001: Auswahl des ML-Anwendungsfalls

## Status
Akzeptiert

## Kontext
Der Prototyp benötigt einen klassischen tabellarischen Machine-Learning-Anwendungsfall, um den vollständigen Lifecycle von Datenaufbereitung über Training bis Monitoring demonstrieren zu können.

## Entscheidung
Als Demonstrationsszenario wird der UCI Bank Marketing Datensatz verwendet.

## Alternativen
- Telco Customer Churn
- Kreditrisikobewertung
- Fraud Detection
- Nachfrageprognose

## Begründung
Der UCI Bank Marketing Datensatz ist öffentlich verfügbar, tabellarisch, überschaubar und für binäre Klassifikation geeignet. Damit eignet er sich gut für reproduzierbare Trainingspipelines, Modellversionierung, API-Serving und Drift-Simulation.

## Konsequenzen
Die fachliche Story ist weniger intuitiv als bei Churn Prediction, dafür ist der Datensatz wissenschaftlich sauberer und stabiler verwendbar. 
# Lightweight MLOps Prototype

A locally executable MLOps prototype for reproducible model training,
model versioning, API-based serving and batch drift monitoring.

## Motivation

Machine-learning models are often initially developed in notebooks.
Moving from an experiment to a reproducible and monitorable system
requires additional mechanisms for pipeline automation, traceability,
model management, deployment and monitoring.

This project demonstrates a lightweight implementation of this
lifecycle for a resource-constrained local development environment.

## Machine-Learning Use Case

The prototype uses the UCI Bank Marketing dataset.

The binary classification task predicts whether a bank customer will
subscribe to a term deposit.

## Architecture

```mermaid
flowchart LR
    A[Raw Data] --> B[Data Preparation]
    B --> C[Model Training]
    C --> D[Model Evaluation]
    D --> E[MLflow Tracking]
    E --> F[Model Registry]
    F --> G[FastAPI Prediction Service]
    B --> H[Simulated Current Data]
    C --> I[Prediction Probabilities]
    H --> J[Evidently Drift Monitoring]
    I --> J
# System Architecture

## Overview

The Rural Emergency Triage AI Assistant is built on a modular, offline-first architecture designed for deployment in resource-constrained environments.

## Components

### 1. Imaging Analysis Module
- **Model**: MedGemma 1.5 4B (fine-tuned)
- **Input**: DICOM files or PNG/JPEG images
- **Output**: Critical findings with bounding boxes and confidence scores

### 2. Speech Recognition Module
- **Model**: MedASR
- **Input**: Clinical dictation (WAV/MP3)
- **Output**: Structured clinical notes

### 3. Decision Support Engine
- **Logic**: Rule-based + AI recommendations
- **Output**: Triage priority, transfer recommendations

### 4. User Interface
- **Platform**: React Native (iOS/Android)
- **Mode**: Offline-first with optional cloud sync

## Data Flow
```
[Clinician] → [Voice Input] → [MedASR] → [Structured Text]
              [Image Upload] → [MedGemma] → [Critical Findings]
                                        ↓
                              [Decision Support Engine]
                                        ↓
                              [Triage Recommendation] → [Display]
```

## Deployment Architecture

- **Edge Device**: Hospital tablet (8GB RAM, ARM/x86)
- **Backend**: FastAPI (local server on device)
- **Models**: Quantized to 4-bit for efficiency
- **Storage**: Local SQLite database
- **Sync**: Optional background upload when connected


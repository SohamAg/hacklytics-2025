# FairCall AI - Hackalytics 2025
Project for hackytics 2025

FairCall AI aimed to analyze NFL penalty(pass interference)footage by extracting frames, detecting key objects such as people and moving elements using YOLO, and classifying interactions using LTSM. Additionally, an LLM was used to generate explanations for whether a pass interference call should be made. The final product provided an automated system to identify playres, track movement, recognize physical interactions, and generate explanations for key events based on the detected patterns. 

## Features

- **LTSM** - this was used to classify interactions and differentiate between different types of pass interference penalties like offensive or defensive
- **LLM** - this helps explain why the specific call for the frame was made to justify the rationale behind the LTSM decision to make the judgement
- **YOLO** - this was used to automate object detection, and convert frames into coordinates/numbers to easily identify patterns

-  **How It Works**
FairCall AI was created to help fairly and accurately produce correct officating calls for football's most controversial/important call 
1. Frame Extraction & Object Detection - We extracted frames from DPI and OPI videos and used YOLO to detect key objects like players and the ball, as well as track important interactions.
2. LTSM Classification - We applied LTSM algorithms to classify different play types, distinguishing between DPI and OPI, to ensure accurate event recognition.
3. LLM Explanation System - An integrated language model (LLM) generates explanations for key events, such as justifying pass interference calls based on the detected interactions. 

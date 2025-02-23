# FairCall AI - Hackalytics 2025


**FairCall AI** is aimed to analyze NFL penalty(pass interference)footage by extracting frames and detecting key objects such as people and moving elements using YOLO. Additionally, an LLM was used to generate explanations for whether a pass interference call should be made. The final product provides an automated system to identify players, track movement, recognize physical interactions, and generate explanations for key events based on the detected patterns. 

## Features

- **LLM** - this helps explain why the specific call for the frame was made to justify the rationale behind the LTSM decision to make the judgement
- **YOLO** - this was used to automate object detection, and convert frames into coordinates/numbers to easily identify patterns

 ## How It Works
  
FairCall AI was created to help fairly and accurately produce correct officiating calls for football's most controversial/important call. 
1. **Frame Extraction & Object Detection** - We extracted frames from DPI and OPI videos and used YOLO to detect key objects like players and the ball, as well as track important interactions.
3. **LLM Explanation System** - An integrated language model (LLM) generates explanations for key events, such as justifying pass interference calls based on the detected interactions.

## Technology Stack

- **Frontend:** Built with Streamlit user-friendly experience
- **Backend:** The backend is primarily Python, passing information between LM Studio and frontend
- **LLM**: used LM studio along with Python scripting to provide reasoning for penalty calls and plays
- **YOLO**: YOLO detects objects in frame and has been trained on players and footballs

## Getting Started 
To begin using FairCall AI:
1. Clone this repository and install the necessary dependencies
2. Ensure you have access to the video footage (DPI and OPI) and load them into the system.
3. Run the script to detect key events using YOLO and classify interactions.
4. Set up LM Studio with the InternLM 20B model.
5. Get real-time explanations of critical sports moments, emphasizing pass interference calls, generated by the integrated LLM.

## License 

FairCall AI is licensed under the [MIT License](LICENSE.txt). Feel free to use, modify, and distribute this software in accordance with the license terms.

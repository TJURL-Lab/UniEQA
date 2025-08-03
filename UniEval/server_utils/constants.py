CONTROLLER_HEART_BEAT_EXPIRATION = 90
WORKER_HEART_BEAT_INTERVAL = 30
WORKER_API_TIMEOUT = 600 # 20

LOGDIR = "/home/fx/Exp2/test/EmbodiedEval/log"
CONVERSATION_SAVE_DIR = '/home/fx/Exp2/test/EmbodiedEval/conversation_log'


rules_markdown = """ ## 📜 Rules
- Load an image and ask any question to two anonymous models and vote for the better one!
- You can continue chatting until you identify a winner.
- Two models are anonymous before your vote.
- Click “Clear history” to start a new round.
## 👇 Chat now!
"""

# - [[GitHub]](https://github.com/OpenGVLab/Multi-modality-Arena)

# rules_markdown = """ ### Rules
# - Vote for large multi-modality models on visual question answering.
# - Load an image and ask a question. Only one question is supported per round.
# - Two models are anonymous before your vote.
# - Click “Clear history” to start a new round.
# - [[GitHub]](https://github.com/OpenGVLab/Multi-modality-Arena)
# """


# notice_markdown = """ ### Terms of use
# By using this service, users are required to agree to the following terms: The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. **The service collects user dialogue data and reserves the right to distribute it under a Creative Commons Attribution (CC-BY) license.** The demo works better on desktop devices with a wide screen.
# """
notice_markdown = """ ### Terms of Use
Users are required to agree to the following terms before using the service:

The service is a research preview. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. Please do not upload any private information. The service collects user dialogue data, including both text and images, and reserves the right to distribute it under a Creative Commons Attribution (CC-BY) or a similar license.
"""


license_markdown = """ ### License
This project follows the Apache-2.0 license. The service is a research preview intended for non-commercial use only. Please contact us if you find any potential violation.
"""
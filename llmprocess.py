import os
import io
from pydantic import BaseModel
import openai
from openai import OpenAI
import sqlite3

openai.api_base = "http://127.0.0.1:1234"
openai.api_key = ""
client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="unneeded")

def get_pi(var1, var2, var3, var4, var5):
    sys_msg_file = open("piprompt.txt")
    sys_msg = sys_msg_file.read()
    sys_msg_file.close()

    completion = client.chat.completions.create(
        model="internlm/internlm2_5-20b-chat-gguf/",
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": var1 + ", " + var2 + ", " + var3 + ", " + var4 + ", " + var5}
        ],
    )

    eval = completion.choices[0].message.content
    output_to_site(eval)


def output_to_site(eval):
    print(eval)


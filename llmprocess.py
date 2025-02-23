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









# conn = sqlite3.connect("HackathonDB.db")

# cur = conn.cursor()
# cur2 = conn.cursor()

# for row in cur.execute('SELECT * FROM data'):
#     #read row
#     #separate by field
#     obsv = str(row[0])
#     datetm = str(row[1])
#     pnt = str(row[2])
#     qual = str(row[3])
#     atn = str(row[4])
#     atn_f = str(row[5])
    
    
#     completion = client.chat.completions.create(
#         model="internlm/internlm2_5-20b-chat-gguf/internlm2_5-20b-chat-q4_0.gguf",
#         messages=[
#             {"role": "system", "content": sys_msg},
#             {"role": "user", "content": pnt + ", " + qual + ", " + atn + ". " + atn_f}
#         ],
#     )

#     eval = completion.choices[0].message.content

#     #print(eval)

#     mdl = eval.split()

#     psif = mdl[0]
#     category = mdl[1]
#     confidence = mdl[2]

#     #print(psif)
#     #print(category)
#     #print(confidence)
#     #print(obsv)

#     cur2.execute("UPDATE data SET PSIF =" + psif + ", CATEGORY =" + category + ", CONFIDENCE =" + confidence + " WHERE OBSRVTN_NB =" + obsv)
#     conn.commit() 
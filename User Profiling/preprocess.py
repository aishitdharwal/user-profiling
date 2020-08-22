import json

everything = json.loads(open('data.json').read())

print(everything[0]['content'][0])

print(everything[0])
print(everything[1])

everything[0]['time']

print(len(everything))

content=[]
for i in range(len(everything)):
    content=content+list(everything[i]['content'])

len(content)

for i in range(len(everything)):
    del everything[i]['time'][0]

time=[]
for i in range(len(everything)):
    time=time+list(everything[i]['time'])

from datetime import datetime
t1=datetime.strptime('May 17, 2019, 3:40:43 AM UTC', '%b %d, %Y, %I:%M:%S %p %Z')
t2=datetime.strptime('May 17, 2019, 3:40:43 PM UTC', '%b %d, %Y, %I:%M:%S %p %Z')

print(t2-t1)

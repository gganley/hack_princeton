import json
import requests

ffz = requests.get('https://api.frankerfacez.com/v1/emoticons?sort=count-desc').json().get('emoticons')
frankerfacez = {}
for x in ffz:
    frankerfacez[x['name']] = '<img src="https://' + x['urls']['1'][2:] + '">'


def process():
    with open("/Users/gganley/Downloads/imaqtpie/v333193994.json", "r") as read_file:
        data = json.load(read_file)
        print(data)


def replace_sub_with(fragments):
    if len(fragments) == 0:
        return ""
    if 'emoticon' in fragments[0].keys():
        return '<img src="https://static-cdn.jtvnw.net/emoticons/v1/' + fragments[0].get('emoticon').get('emoticon_id') + '/1.0"> ' + replace_sub_with(fragments[1:])
    elif 'text' in fragments[0].keys():
        return replace_with_ffz(fragments[0].get('text')) + replace_sub_with(fragments[1:])



def run_thing():
    data = []
    with open("/Users/gganley/Downloads/imaqtpie/v333193994.json", "r") as read_file:
        data = json.load(read_file)
    this_data = []
    test = ""
    for x in data.get('comments'):
        this_data.append(x['message']['fragments'])

    for x in this_data:
        test += "<p>" + replace_sub_with(x) + "<br></p>\n"
    with open('/Users/gganley/this_thing.html', 'w') as write_file:
        write_file.write("""
        <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Title</title>
</head>
<body>
""" + test + """
</body>
</html>""")


def replace_with_ffz(test_string):
    retstring = ""
    entries = list(filter(lambda x: x in test_string, frankerfacez.keys()))
    if len(entries) != 0:
        for entry in entries:
            retstring += frankerfacez[entry] + ' '
        return retstring
    else:
        return ''

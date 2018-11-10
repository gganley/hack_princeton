import json
def process():
    with open("/Users/gganley/Downloads/imaqtpie/v333193994.json", "r") as read_file:
        data = json.load(read_file)
        print(data)


def replace_sub_with(fragments):
    if len(fragments) == 0:
        return ""
    if 'emoticon' in fragments[0].keys():
        return '<img src="https://static-cdn.jtvnw.net/emoticons/v1/' + fragments[0].get('emoticon').get('emoticon_id') + '/1.0"> ' + replace_sub_with(fragments[1:])
    else:
        return fragments[0].get("text") + " " + replace_sub_with(fragments[1:])

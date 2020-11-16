import requests

CHAT_ID = ''
TELEGRAM_BOT_TOKEN = ''

url = 'https://api.telegram.org/bot'+TELEGRAM_BOT_TOKEN+'/sendMessage'

def sendMsg(*msg):
    try:
        alertMsg = ' '.join(map(str,msg))
        body = {'chat_id': CHAT_ID, 'text': str(alertMsg) }
        x = requests.post(url, data = body)
        print('Done sending message:', alertMsg)
        print('status code:', x.status_code)

    except Exception as e:
        print('ERROR:', e)

        

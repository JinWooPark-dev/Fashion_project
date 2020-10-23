import os
import sys
import urllib.request

# 실행 시키기 전 voice폴더 생성
# right_trash = 쓰레기 통, throw_trash = 버려진 쓰레기
def voice(right_trash, throw_trash):
    right_trash = right_trash
    throw_trash = throw_trash
    client_id = "109vqxa5ud"
    client_secret = "z5s6MESERmZmNqhCv3TyXkudZfEfFpzPOrI2y1PB"
    if right_trash != throw_trash:
        encText = urllib.parse.quote(f"지금 넣으신 쓰레기는 {right_trash}이 아니라 {throw_trash}입니다. {throw_trash} 통에 넣어주세요.")
    else:
        return
    data = "speaker=mijin&speed=0&text=" + encText;
    url = "https://naveropenapi.apigw.ntruss.com/tts-premium/v1/tts"
    request = urllib.request.Request(url)
    request.add_header("X-NCP-APIGW-API-KEY-ID",client_id)
    request.add_header("X-NCP-APIGW-API-KEY",client_secret)
    response = urllib.request.urlopen(request, data=data.encode('utf-8'))
    rescode = response.getcode()
    if(rescode==200):
        print("TTS mp3 저장")
        response_body = response.read()
        with open('./voice/1234.mp3', 'wb') as f:
            f.write(response_body)
    else:
        print("Error Code:" + rescode)

if __name__ == '__main__':
    voice("플라스틱", "캔")
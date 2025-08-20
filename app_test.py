import json
#
from app import app


apptestclient = t = app.test_client()


def runtest_ask_Nquestion():
  test_questions = [
    'thông tin về trường',
    'giới thiệu trường',
    'học phí',
    'chương trình đào tạo',
    'thông tin tuyển sinh',
    'cơ sở vật chất',
  ]

  for question in test_questions:
    test_data = {"question": question}
    print(f'\n{test_data=!s}\n')
    runtest_ask_1question(test_data)

def runtest_ask_1question(test_data):
  res = t.post('/ask', data=json.dumps(test_data), content_type='application/json')

  print(f'{res.status_code=!s}')
  print(f'{res.data=!s}')

  assert res.status_code in [200, 404]

  d = json.loads(res.data)
  assert 'response'   in d
  assert 'media_type' in d
  assert 'images'     in d
  assert 'captions'   in d
  assert 'video_url'  in d

if __name__ == '__main__':
  runtest_ask_1question(test_data = {"question": 'thông tin về trường' })
  # runtest_ask_Nquestion()

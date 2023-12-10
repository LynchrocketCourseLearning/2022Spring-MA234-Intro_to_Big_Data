import requests

def save_file(respond):
    with open('spider_text.html','w',encoding='utf-8') as f:
        f.write(respond.text)
        print('save success!!')

class Spider:
    def __init__(self, url, word=None, page=1):
        self.url = url
        self.word = word
        self.page = page

    def get(self):
        respond = requests.get(self.url)
        save_file(respond)
        return respond
    
    def get_parse(self, respond):
        return respond

    def post(self):
        form_data = {
            'i': self.word,
            'from': 'AUTO',
            'to': 'AUTO',
            'smartresult': 'dict',
            'client': 'fanyideskweb',
            'salt': '16455804091741',
            'sign': '644a2fd4a86fcc900283d37ba6b26df9',
            'lts': '1645580409174',
            'bv': '56d33e2aec4ec073ebedbf996d0cba4f',
            'doctype': 'json',
            'version': '2.1',
            'keyfrom': 'fanyi.web',
            'action': 'FY_BY_REALTlME'
        }
        reqspond = requests.post(self.url,data=form_data)
        print(reqspond.txt)
        return reqspond

    def get_like_post(self):
        url = self.url.format(self.page, self.word)
        res = requests.get(url)
        save_file(res)


Url = ['http://www.cntour.cn/',
        'https://fanyi.youdao.com/translate?smartresult=dict&smartresult=rule',
        'https://so.gushiwen.cn/search.aspx?type=author&page={0}&value={1}'
        ]
# get
# spider = Spider(Url[0])
# spider.get()

# post
# Word = input('请输入翻译的内容：')
# spider = Spider(Url[1], Word)
# Spider.post()

# get_like_post
Word = input('请输入搜索的内容：')
Page = input('请输入搜索的页码：')
spider = Spider(Url[2], Word, int(Page))
spider.get_like_post()
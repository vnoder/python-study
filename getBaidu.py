import requests


def get_baidu():
    url = 'https://www.baidu.com/'
    r = requests.get(url)
    print(r)
    print(r.text)


if __name__ == '__main__':
    get_baidu()
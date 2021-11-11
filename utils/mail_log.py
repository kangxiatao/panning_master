# -*- coding: utf-8 -*-

"""
Created on 08/22/2021
mail_log.
@author: Kang Xiatao (kangxiatao@gmail.com)
"""

# --- 邮箱部分 ---
from email.header import Header
from email.mime.text import MIMEText
from email.utils import parseaddr, formataddr
import smtplib

import requests
import random
import re


def validateEmail(email):

    if len(email) > 7:
        if re.match("^.+\\@(\\[?)[a-zA-Z0-9\\-\\.]+\\.([a-zA-Z]{2,3}|[0-9]{1,3})(\\]?)$", email) != None:
            return 1
    return 0


def _format_addr(s):
    name, addr = parseaddr(s)
    return formataddr((Header(name, 'utf-8').encode(), addr))


def get_sentence():
    # 获取金山词霸每日一句，英文和翻译
    url = "http://open.iciba.com/dsapi/"
    r = requests.get(url)
    # print(r.json())
    contents = r.json()['content']
    note = r.json()['note']
    translation = r.json()['translation']
    print(contents, note)
    return contents + '\n' + note


def get_words():
    res = ['Wubba lubba dub dub', 'haha']
    try:
        con = requests.get('http://api.eei8.cn/say/api.php?encode=js')
        # print(con.content.decode('utf-8'))
        res = re.findall("document.write\('(.*?)'\);", con.content.decode('utf-8'))
        print(res[0])

        # con = requests.get('https://v1.hitokoto.cn')
        # con = requests.get('https://international.v1.hitokoto.cn')
        # print(con.content.decode('utf-8'))
        # res = re.findall("\"hitokoto\":\"'(.*?)'\";", con.content.decode('utf-8'))
        # print(res[0])

    except:
        res = ['Wubba lubba dub dub', 'haha']
        print(res[0])

    return res[0]


class MailLogs:
    # 使用第三方 SMTP 服务
    # 在邮箱设置中开启SMTP服务，并记住password
    def __init__(self, from_addr='2072483140@qq.com', to_addr='799802172@qq.com',
                 password='mrzmvxtvbpvjbief', smtp_server='smtp.qq.com'):
        self.from_addr = from_addr
        self.to_addr = to_addr
        self.password = password
        self.smtp_server = smtp_server

    def sendmail(self, sendstr, fromstr='炼丹炉', header='我能在河边钓一整天的🐟'):
        try:
            msg = MIMEText(sendstr, 'plain', 'utf-8')
            msg['From'] = _format_addr(fromstr + '<%s>' % self.from_addr)
            msg['To'] = _format_addr('<%s>' % self.to_addr)
            msg['Subject'] = Header(header, 'utf-8').encode()

            server = smtplib.SMTP_SSL(self.smtp_server, 465)
            server.ehlo(self.smtp_server)  # linux报错解决
            # server.set_debuglevel(1)  # 打印出和SMTP服务器交互的所有信息
            server.login(self.from_addr, self.password)
            server.sendmail(self.from_addr, [self.to_addr], msg.as_string())
            server.quit()
        except:
            print(sendstr, fromstr, header)

    def set_to(self, to_addr):
        self.to_addr = to_addr


if __name__ == '__main__':

    import time

    QQmail = MailLogs()
    QQmail.sendmail('test', header=get_words())

    while 1:
        current_time = time.localtime(time.time())
        if current_time.tm_hour == 23 and current_time.tm_min == 54 and current_time.tm_sec == 10:
            print("test success")
        if current_time.tm_hour == 8 and current_time.tm_min == 0 and current_time.tm_sec == 0:
            QQmail.set_to('799802172@qq.com')
            QQmail.sendmail(get_sentence(), header=get_words())
            time.sleep(1)
            QQmail.set_to('983790602@qq.com')
            QQmail.sendmail(get_sentence(), header=get_words())
            time.sleep(1)
            QQmail.set_to('1577215303@qq.com')
            QQmail.sendmail(get_sentence(), header=get_words())
        # if current_time.tm_hour == current_time.tm_min and current_time.tm_hour == current_time.tm_sec:
        #     QQmail.set_to('799802172@qq.com')
        #     QQmail.sendmail(get_words())
        #     time.sleep(1)
        #     QQmail.set_to('983790602@qq.com')
        #     QQmail.sendmail(get_words())
        #     time.sleep(1)
        #     QQmail.set_to('1577215303@qq.com')
        #     QQmail.sendmail(get_words())
        time.sleep(1)


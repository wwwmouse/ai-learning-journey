import requests as rq
import sys
# def get_followers(username):
#     r = rq.get(f"https://api.github.com/users/{username}")
#     if r.status_code == 200:
#         data=r.json()
#         return data['followers']
#     else:
#         return None

# name=input("请输入用户名:")
# print(f"{get_followers(name)}")

class githubUser:
    def __init__(self,username):
        self.username=username
    def get_follower(self):
        r = rq.get(f"https://api.github.com/users/{self.username}",timeout=10)
        if r.status_code == 200:
            data=r.json()
            return data['followers']
        else:
            return None
        
name=sys.argv[1]      
test=githubUser(name)
print(f"粉丝数：{test.get_follower()}")
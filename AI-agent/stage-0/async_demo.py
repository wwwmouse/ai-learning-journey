import asyncio,time

# async def say_after(seconds,msg):
#     await asyncio.sleep(seconds)
#     print(msg)
    
# async def main():
#     t=time.time()
#     await asyncio.gather(
#         say_after(2,"A finish"),
#         say_after(2,"B finish"),
#     )
#     print(f"总耗时 {round(time.time() - t, 1)} 秒")
# asyncio.run(main())

def sync_version():
    t = time.time()
    time.sleep(2)        # 堵住门睡 2 秒
    print("A 完成")
    time.sleep(2)        # 再堵 2 秒
    print("B 完成")
    print(f"总耗时 {round(time.time() - t, 1)} 秒")

sync_version()
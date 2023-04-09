import time
import requests
from tqdm import tqdm
import os


cookies = {
    'BDqhfp': '%E7%8B%97%E7%8B%97%26%26NaN-1undefined%26%2618880%26%2621',
    'BIDUPSID': '06338E0BE23C6ADB52165ACEB972355B',
    'PSTM': '1646905430',
    'BAIDUID': '104BD58A7C408DABABCAC9E0A1B184B4:FG=1',
    'BDORZ': 'B490B5EBF6F3CD402E515D22BCDA1598',
    'H_PS_PSSID': '35836_35105_31254_36024_36005_34584_36142_36120_36032_35993_35984_35319_26350_35723_22160_36061',
    'BDSFRCVID': '8--OJexroG0xMovDbuOS5T78igKKHJQTDYLtOwXPsp3LGJLVgaSTEG0PtjcEHMA-2ZlgogKK02OTH6KF_2uxOjjg8UtVJeC6E\
        G0Ptf8g0M5',
    'H_BDCLCKID_SF': 'tJPqoKtbtDI3fP36qR3KhPt8Kpby2D62aKDs2nopBhcqEIL4QTQM5p5yQ2c7LUvtynT2KJnz3Po8MUbSj4QoDjFjXJ7RJR\
        JbK6vwKJ5s5h5nhMJSb67JDMP0-4F8exry523ioIovQpn0MhQ3DRoWXPIqbN7P-p5Z5mAqKl0MLPbtbb0xXj_0D6bBjHujtT_s2TTKLPK8fCn\
        BDP59MDTjhPrMypomWMT-0bFH_-5L-l5js56SbU5hW5LSQxQ3QhLDQNn7_JjOX-0bVIj6Wl_-etP3yarQhxQxtNRdXInjtpvhHR38MpbobUPU\
        Da59LUvEJgcdot5yBbc8eIna5hjkbfJBQttjQn3hfIkj0DKLtD8bMC-RDjt35n-Wqxobbtof-KOhLTrJaDkWsx7Oy4oTj6DD5lrG0P6RHmb8h\
        t59JROPSU7mhqb_3MvB-fnEbf7r-2TP_R6GBPQtqMbIQft20-DIeMtjBMJaJRCqWR7jWhk2hl72ybCMQlRX5q79atTMfNTJ-qcH0KQpsIJM5-\
        DWbT8EjHCet5DJJn4j_Dv5b-0aKRcY-tT5M-Lf5eT22-usy6Qd2hcH0KLKDh6gb4PhQKuZ5qutLTb4QTbqWKJcKfb1MRjvMPnF-tKZDb-JXtr\
        92nuDal5TtUthSDnTDMRhXfIL04nyKMnitnr9-pnLJpQrh459XP68bTkA5bjZKxtq3mkjbPbDfn02eCKuj6tWj6j0DNRabK6aKC5bL6rJabC3\
        b5CzXU6q2bDeQN3OW4Rq3Irt2M8aQI0WjJ3oyU7k0q0vWtvJWbbvLT7johRTWqR4enjb3MonDh83Mxb4BUrCHRrzWn3O5hvvhKoO3MA-yUKmD\
        loOW-TB5bbPLUQF5l8-sq0x0bOte-bQXH_E5bj2qRCqVIKa3f',
    'BDSFRCVID_BFESS': '8--OJexroG0xMovDbuOS5T78igKKHJQTDYLtOwXPsp3LGJLVgaSTEG0PtjcEHMA-2ZlgogKK02OTH6KF_2uxOjjg8UtVJ\
        eC6EG0Ptf8g0M5',
    'H_BDCLCKID_SF_BFESS': 'tJPqoKtbtDI3fP36qR3KhPt8Kpby2D62aKDs2nopBhcqEIL4QTQM5p5yQ2c7LUvtynT2KJnz3Po8MUbSj4QoDjFjX\
        J7RJRJbK6vwKJ5s5h5nhMJSb67JDMP0-4F8exry523ioIovQpn0MhQ3DRoWXPIqbN7P-p5Z5mAqKl0MLPbtbb0xXj_0D6bBjHujtT_s2TTKLP\
        K8fCnBDP59MDTjhPrMypomWMT-0bFH_-5L-l5js56SbU5hW5LSQxQ3QhLDQNn7_JjOX-0bVIj6Wl_-etP3yarQhxQxtNRdXInjtpvhHR38Mpb\
        obUPUDa59LUvEJgcdot5yBbc8eIna5hjkbfJBQttjQn3hfIkj0DKLtD8bMC-RDjt35n-Wqxobbtof-KOhLTrJaDkWsx7Oy4oTj6DD5lrG0P6R\
        Hmb8ht59JROPSU7mhqb_3MvB-fnEbf7r-2TP_R6GBPQtqMbIQft20-DIeMtjBMJaJRCqWR7jWhk2hl72ybCMQlRX5q79atTMfNTJ-qcH0KQps\
        IJM5-DWbT8EjHCet5DJJn4j_Dv5b-0aKRcY-tT5M-Lf5eT22-usy6Qd2hcH0KLKDh6gb4PhQKuZ5qutLTb4QTbqWKJcKfb1MRjvMPnF-tKZDb\
        -JXtr92nuDal5TtUthSDnTDMRhXfIL04nyKMnitnr9-pnLJpQrh459XP68bTkA5bjZKxtq3mkjbPbDfn02eCKuj6tWj6j0DNRabK6aKC5bL6r\
        JabC3b5CzXU6q2bDeQN3OW4Rq3Irt2M8aQI0WjJ3oyU7k0q0vWtvJWbbvLT7johRTWqR4enjb3MonDh83Mxb4BUrCHRrzWn3O5hvvhKoO3MA-\
        yUKmDloOW-TB5bbPLUQF5l8-sq0x0bOte-bQXH_E5bj2qRCqVIKa3f',
    'indexPageSugList': '%5B%22%E7%8B%97%E7%8B%97%22%5D',
    'cleanHistoryStatus': '0',
    'BAIDUID_BFESS': '104BD58A7C408DABABCAC9E0A1B184B4:FG=1',
    'BDRCVFR[dG2JNJb_ajR]': 'mk3SLVN4HKm',
    'BDRCVFR[-pGxjrCMryR]': 'mk3SLVN4HKm',
    'ab_sr': '1.0.1_Y2YxZDkwMWZkMmY2MzA4MGU0OTNhMzVlNTcwMmM2MWE4YWU4OTc1ZjZmZDM2N2RjYmVkMzFiY2NjNWM4Nzk4NzBlZTliYWU0ZT\
        AyODkzNDA3YzNiMTVjMTllMzQ0MGJlZjAwYzk5MDdjNWM0MzJmMDdhOWNhYTZhMjIwODc5MDMxN2QyMmE1YTFmN2QyY2M1M2VmZDkzMjMyOTh\
        iYmNhZA==',
    'delPer': '0',
    'PSINO': '2',
    'BA_HECTOR': '8h24a024042g05alup1h3g0aq0q',
}
headers = {
    'Connection': 'keep-alive',
    'sec-ch-ua': '" Not;A Brand";v="99", "Google Chrome";v="97", "Chromium";v="97"',
    'Accept': 'text/plain, */*; q=0.01',
    'X-Requested-With': 'XMLHttpRequest',
    'sec-ch-ua-mobile': '?0',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) \
        Chrome/97.0.4692.99 Safari/537.36',
    'sec-ch-ua-platform': '"macOS"',
    'Sec-Fetch-Site': 'same-origin',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Dest': 'empty',
    'Referer': 'https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=result&fr\
    =&sf=1&fmq=1647837998851_R&pv=&ic=&nc=1&z=&hd=&latest=&copyright=&se=1&showtab=0&fb=0&width=&height=&face=0&istype\
    =2&dyTabStr=MCwzLDIsNiwxLDUsNCw4LDcsOQ%3D%3D&ie=utf-8&sid=&word=%E7%8B%97%E7%8B%97',
    'Accept-Language': 'zh-CN,zh;q=0.9',
}


# 定义函数进行图片爬取
def craw_single_class(keyword, download_num=200):
    if os.path.exists('dataset/' + keyword):
        print('文件夹 dataset/{} 已存在，之后直接将爬取到的图片保存至该文件夹中'.format(keyword))
    else:
        os.makedirs('dataset/{}'.format(keyword))
        print('新建文件夹：dataset/{}'.format(keyword))
    count = 1

    with tqdm(total=download_num, position=0, leave=True) as pbar:

        # 爬取第几张
        num = 0

        # 是否继续爬取
        flag = True

        while flag:

            page = 30 * count

            params = (
                ('tn', 'resultjson_com'),
                ('logid', '12508239107856075440'),
                ('ipn', 'rj'),
                ('ct', '201326592'),
                ('is', ''),
                ('fp', 'result'),
                ('fr', ''),
                ('word', f'{keyword}'),
                ('queryWord', f'{keyword}'),
                ('cl', '2'),
                ('lm', '-1'),
                ('ie', 'utf-8'),
                ('oe', 'utf-8'),
                ('adpicid', ''),
                ('st', '-1'),
                ('z', ''),
                ('ic', ''),
                ('hd', ''),
                ('latest', ''),
                ('copyright', ''),
                ('s', ''),
                ('se', ''),
                ('tab', ''),
                ('width', ''),
                ('height', ''),
                ('face', '0'),
                ('istype', '2'),
                ('qc', ''),
                ('nc', '1'),
                ('expermode', ''),
                ('nojc', ''),
                ('isAsync', ''),
                ('pn', f'{page}'),
                ('rn', '30'),
                ('gsm', '1e'),
                ('1647838001666', ''),
            )

            response = requests.get('https://image.baidu.com/search/acjson', headers=headers, params=params,
                                    cookies=cookies)
            if response.status_code == 200:
                try:
                    json_data = response.json().get("data")

                    if json_data:
                        for x in json_data:
                            type = x.get("type")
                            if type not in ["gif"]:
                                img = x.get("thumbURL")
                                fromPageTitleEnc = x.get("fromPageTitleEnc")
                                try:
                                    resp = requests.get(url=img, verify=False)
                                    time.sleep(1)
                                    # print(f"链接 {img}")

                                    # 保存文件名
                                    # file_save_path = f'dataset/{keyword}/{num}-{fromPageTitleEnc}.{type}'
                                    file_save_path = f'dataset/{keyword}/{num}.{type}'
                                    with open(file_save_path, 'wb') as f:
                                        f.write(resp.content)
                                        f.flush()
                                        # print('第 {} 张图像 {} 爬取完成'.format(num, fromPageTitleEnc))
                                        num += 1
                                        pbar.update(1)  # 进度条更新

                                    # 爬取数量达到要求
                                    if num > download_num:
                                        flag = False
                                        print('{} 张图像爬取完毕'.format(num))
                                        break

                                except Exception:
                                    pass
                except:
                    pass
            else:
                break

            count += 1


# 爬取一类
craw_single_class('柚子', download_num=10)
# 爬取多类
class_list = ['黄瓜', '南瓜', '冬瓜']
for each in class_list:
    craw_single_class(each, download_num=10)

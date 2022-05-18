from tensorflow.keras import backend as K
import numpy as np
import argparse
import torch
from torch.autograd import Variable
import torchvision
from torchvision.ops import nms
from PIL import Image
import numpy as np
from skimage.draw import rectangle_perimeter

ALPHABET = u"耳夢建早样事波球成外畫局嘉主但阳道諸乾或路故長的灵勿御岸去部杜四電高苦番方切威聲曰聚班端數始瘦床水遺魚烏武重沉轉清傳觉油火手鄉地此底鳥收貴偏計起用當号弄件亦深運是包代妙艮良庭亮英沙還竟语碎其孝华幽斷身永和表機国離官着連背樓名多微父刀开首化絕業體夏峰易餘無当请洞零白许利野入晨美亭蘇東繁别见土兴葉尊思走似理昔管源要吾相為次期船亡步帶氏阿拉进今果姐逐場窮罗君近色驚强万字命没力客盛將处特雨祖久暗蓮轻遠范位睡侯群求可食降龍樹肉米裡飛些病存射季途後把不梁眼雲条親回末商系屋智寒時散這關乃浩待家卷赤殺布雷蘭里虛雪華甚春頭終勢著勝村黃眠臨宏府累矣影车吹宝说基孫盤女段窗候旅爱傾比齊湖心圖服遊先寄初足月種各紅能付除甘題書声露桂消聞界草往芳穿調东算登獨右会洗玉式遇言休六景慶羽乘應識出破澤鳴乎奈辰很等们复德孤放沈一節論真海柳千夂宜叫瑞蛋彩个看日校傷車說惠修血无性子精味垂集克指害豐靜義便政短報唐九羅也復圓吃軍合張考康列敬台刻皇語師臺悠送业社塞曾號角漢溪梅悲加活陰套住發喜胡宇百魂俊科欠正限歸健石又問配夫平豪觀毛任夕施森昭晴已周河淡徐乐目下逆那过既醉極伯排菜汝田将中室人查止楚仁守使世未誰到承寂友干惡钱氣童昌哀陽忽向二最口戰泉你而以受半急分博榮珍灰息敢院想在緣狂伊失黎滿第泰鬼征忠得賢章銀左孟空八暖渡樂隔來禮臣謝金喝輕治才就阮志象古男凡疑丹量須李吳衣飲天告奇器間为有张迎與五常川如上難只開像過盡封立宗兒支戈小恨爾細文完則非知顧星懷带兵遍度宿酒愁亲進儿郎冠尚佛含好風还坤对云反般怨吟卒做教牙王從萬佳惟共何舉竹经希聽江許我死感北浮采这大引玄夜者國忍快再格师羊薄学演南幸都居场山照交虎順雄流气西折笑寶谢神愛然異塵发浪望太少及紫元同煙實辛朝并暴他若弟龙杯余信恩从園總秋馬桑明午推差留体造翠彼辭慧关牛猶形池跟宮林聖料自覺面青丁之本予井魔延通壁鹿達直持年情与动机柔意伏筆奉徒司市桃戴門兮鐘骨改皮尔堂申实依己陳暮解翼香興片母符斯聊哥巴燕曲七舊低马蒼沒琴超且雅陈念至越甲詩什朱福省於几点忘記士静所令游會风坐民冷秀姓動行枝新忙鼓停欲藏私法茶店更城靈长物生木禾願假秦昏淫見每髮全示經后蕭餐晚雙益刺吉寸皆黑別了逢被老強程原歌斗欣冰倒移接来却取功舍时松雞富黄即迷定舞莫磨尋红座谷书音舟舒變离蒙寺霜碧咸打州邊幾么间前族提翻結招整致橋歲冬麻典席头仙條具錄花借制数保由公光必隨根抱昨呼容處工映三落帝壽品祥因善于京难素作兩安麗學珠亂橫對房婦十尾勤楊寧"
IMG_WIDTH = 256
IMG_HEIGHT = 64
MODEL_PATH = 'models/best_pred_ocr2.h5'

input_size = 512
output_size = 128
def num_to_label(num):
    ret = ""
    for ch in num:
        if ch == -1:  # CTC Blank
            break
        else:
            ret+=ALPHABET[ch]
    return ret

def decode_output(preds):
    decoded = K.get_value(K.ctc_decode(preds, input_length=np.ones(preds.shape[0])*preds.shape[1], 
                                   greedy=True)[0][0])
    return num_to_label(decoded[0])
def _nms( img, predict, nms_score, iou_threshold):
    
    bbox = list()
    score_list = list()
    im_draw = np.asarray(torchvision.transforms.functional.resize(img, (img.size[1], img.size[0]))).copy()
    
    heatmap=predict.data.cpu().numpy()[0, 0, ...]
    offset_y = predict.data.cpu().numpy()[0, 1, ...]
    offset_x = predict.data.cpu().numpy()[0, 2, ...]
    width_map = predict.data.cpu().numpy()[0, 3, ...]
    height_map = predict.data.cpu().numpy()[0, 4, ...]
    
    
    for j in np.where(heatmap.reshape(-1, 1) >= nms_score)[0]:

        row = j // output_size 
        col = j - row*output_size
        
        bias_x = offset_x[row, col] * (img.size[1] / output_size)
        bias_y = offset_y[row, col] * (img.size[0] / output_size)

        width = width_map[row, col] * output_size * (img.size[1] / output_size)
        height = height_map[row, col] * output_size * (img.size[0] / output_size)

        score_list.append(heatmap[row, col])

        row = row * (img.size[1] / output_size) + bias_y
        col = col * (img.size[0] / output_size) + bias_x

        top = row - width // 2
        left = col - height // 2
        bottom = row + width // 2
        right = col + height // 2

        start = (top, left)
        end = (bottom, right)

        bbox.append([top, left, bottom, right])
        
    _nms_index = torchvision.ops.nms(torch.FloatTensor(bbox), scores=torch.flatten(torch.FloatTensor(score_list)), iou_threshold=iou_threshold)
    
    for k in range(len(_nms_index)):
    
        top, left, bottom, right = bbox[_nms_index[k]]
        
        start = (top, left)
        end = (bottom, right)
        
        rr, cc = rectangle_perimeter(start, end=end,shape=(img.size[1], img.size[0]))
        
        im_draw[rr, cc] = (255, 0, 0)
        
    return im_draw
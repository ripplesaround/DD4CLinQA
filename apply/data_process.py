# coding=utf-8

'''
Author: ripples
Email: ripplesaround@sina.com

date: 2021/5/13 12:31
desc:
'''
import json
import os


class dataset_info():
    def __init__(self,dataset_num,filename,path=None):
        self.dataset_num = dataset_num
        self.path = path
        self.filename = filename
    def print(self):
        print("数据集中存在{changdu}个数据".format(changdu=self.dataset_num))



class data_sample():
    def __init__(self,context,qas):
        self.question_answer = []
        self.context = context
        self.qas = qas
        self.process()
    def process(self):
        for i in range(len(self.qas)):
            if self.qas[i]["is_impossible"]:
                # print("hello")
                continue
            # 一个问题，四个回答
            temp=[]
            temp.append(self.qas[i]["question"])
            for j in range(len(self.qas[i]["answers"])):
                temp.append(self.qas[i]["answers"][j]["text"])
            temp.append(self.qas[i]["id"])
            self.question_answer.append(temp)
    def print_qa(self):
        print(self.question_answer)

def read_json(num = 0):
    filename = 'sample_dev_ans.json'
    pred_filename = "nbest_predictions_.json"
    f = open(filename, 'r', encoding="utf-8")
    f_pred = open(pred_filename,"r",encoding="utf-8")
    content = f.read()
    a = json.loads(content)
    dic = a["data"][0]
    info = dataset_info(
        dataset_num=len(a["data"]),
        filename='sample_dev_ans.json',
        path=os.path.abspath(filename)
    )
    # print(type(a["data"][0]))
    data_ans = data_sample(context=a["data"][num]["paragraphs"][0]["context"],qas=a["data"][num]["paragraphs"][0]["qas"])
    data_pred = f_pred.read()
    data_pred = json.loads(data_pred)

    print("读取dev文件 "+filename)
    print("读取结果文件 "+pred_filename)
    f.close()
    return info, data_ans, data_pred


if __name__ == "__main__":
    info, data_ans,data_pred = read_json()
    # print(data_ans.question_answer[2])
    print(type(data_pred))
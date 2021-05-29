import sys
from tkinter import *
from tkinter import messagebox

import tkinter as ttk
import os
import xlrd
from xlutils.copy import copy
import json
from data_process import *

# Label(login,text="验证码：").grid(row=3,column=0,sticky=E)
# verifyCode=Entry(login)
# verifyCode.grid(row=3,column=1)

# 判断用户是否在用户信息.xls文件中
import data_process
import socket


class server:
    def __init__(self):
        self.train_model = []
        self.cnt = 0

    def append(self, model):
        self.cnt += 1
        self.train_model.append(model)

    def find_model(self):
        return self.train_model[self.cnt - 1]

    def print_cmd(self):
        cmd = "train_model "
        cmd += str(self.cnt - 1)
        cmd += " "
        # train最新的model
        cmd += self.train_model[self.cnt - 1].print_cmd()
        return cmd


server_gui = server()


class User:
    def __init__(self, name, passWord):
        self.name = name
        self.passWord = passWord

    def isInExcel(self, data):
        filename = "user_info.xls"
        excel = xlrd.open_workbook(filename, encoding_override="utf-8")
        sheet = excel.sheets()[0]

        sheet_row_mount = sheet.nrows  # 行数
        sheet_col_mount = sheet.ncols  # 列数

        sheet_name = []
        # 所有用户名信息
        for x in range(1, sheet_row_mount):
            y = 0
            sheet_name.append(sheet.cell_value(x, y))

        for x in sheet_name:
            # 找到用户名
            if (data == x):
                # 已有用户
                return 1
                break;
        # 未注册用户
        return -1

    def isPasswordDirect(self, data, passWord):
        filename = "user_info.xls"
        excel = xlrd.open_workbook(filename, encoding_override="utf-8")
        sheet = excel.sheets()[0]

        sheet_row_mount = sheet.nrows  # 行数
        sheet_col_mount = sheet.ncols  # 列数

        sheet_name = []
        # 所有用户名信息
        for x in range(1, sheet_row_mount):
            y = 0
            sheet_name.append(sheet.cell_value(x, y))
        sheet_passWord = []
        # 所有密码信息
        for x in range(1, sheet_row_mount):
            y = 1
            sheet_passWord.append(sheet.cell_value(x, y))

        for i in range(len(sheet_name)):
            if (data == sheet_name[i]):
                # 记录用户名在数组中的位置
                record = i
                break

        for i in range(len(sheet_passWord)):
            if (passWord == sheet_passWord[i]):
                # 判断用户名位置与密码位置是否相同
                # 以及密码是否与用户信息中的密码一致
                if (i == record):
                    # 密码正确
                    return 1
                    break
        # 密码错误
        return -1

    # 设计用户注册GUI界面
    def registereds(self):
        registered = Tk()
        registered.title('registered')
        registered.geometry('230x185')
        Label(registered, text='用户注册').grid(row=0, column=0, columnspan=2)
        Label(registered, text='用户名：').grid(row=1, column=0, sticky=E)
        names = Entry(registered)
        names.grid(row=1, column=1)
        Label(registered, text='密码：').grid(row=2, column=0, sticky=E)
        passwds = Entry(registered, show='●')
        passwds.grid(row=2, column=1)
        Label(registered, text='确认密码：').grid(row=3, column=0)
        repasswd = Entry(registered, show='●')
        repasswd.grid(row=3, column=1)
        Label(registered, text='手机号：').grid(row=4, column=0, sticky=E)
        phonenum = Entry(registered)
        phonenum.grid(row=4, column=1)
        Label(registered, text='身份证号：').grid(row=5, column=0)
        man = Entry(registered)
        man.grid(row=5, column=1)

        # 判断是否含有特殊符号
        def teshufuhao(input_psd):
            string = "~!@#$%^&*()_+-*/<>,.[]\/?"
            for i in string:
                if i in input_psd:
                    return True
            return False

        def registeredes():
            # 密码长度小于8
            if len(passwds.get()) < 8:
                messagebox.showerror(title='wrong', message='注册失败，密码不应少于8位')
            # 密码不同时含有数字、字母和特殊符号
            elif not (
                    any([x.isdigit() for x in passwds.get()]) and any(
                [x.isalpha() for x in passwds.get()]) and teshufuhao(
                passwds.get())):
                messagebox.showerror(title='wrong', message='注册失败，密码格式错误，必须包括字母和数字以及特殊符号')
            # 两次密码输入不一样
            elif passwds.get() != repasswd.get():
                messagebox.showerror(title='wrong', message='注册失败，两次密码不相同')
            # 手机号不正确
            elif not (phonenum.get().isdigit() and len(phonenum.get()) == 11):
                messagebox.showerror(title='wrong', message='注册失败，请输入正确的11位手机号')
            # 身份证号不正确
            elif len(man.get()) != 18:
                messagebox.showerror(title='wrong', message='注册失败，请输入正确的18位身份证号')
            else:
                messagebox.showinfo(title='successful', message='注册成功！')
                # 将新用户信息存入用户信息文件
                excel = xlrd.open_workbook('user_info.xls')
                sheet = excel.sheets()[0]

                nrow = sheet.nrows  # 文件行数

                wb = copy(excel)
                w_sheet = wb.get_sheet(0)
                # 从数据下一行开始写入新用户信息
                w_sheet.write(nrow, 0, names.get())
                w_sheet.write(nrow, 1, repasswd.get())
                w_sheet.write(nrow, 2, phonenum.get())
                w_sheet.write(nrow, 3, man.get())

                wb.save('user_info.xls')

        Button(registered, text='注册', command=registeredes).grid(row=6, column=0, columnspan=3)

    def successful(self):
        # 判断用户名不存在
        msg = 'login %s' % (name.get() +" "+ passWord.get())
        s.send(msg.encode('utf-8'))
        # 接收服务器传回的判断结果
        msg = s.recv(1024)
        result = msg.decode('utf-8').split(" ")
        result_isinexcel = result[0]
        result_ispassworddirect = result[-1]

        if result_isinexcel == -1:
            messagebox.showerror(title='wrong', message='用户不存在，请注册')
        # 密码不正确
        elif result_ispassworddirect == -1:
            messagebox.showerror(title='wrong', message='密码不正确')
        # # 验证码位数不正确
        # elif len(verifyCode.get())!=4:
        #     messagebox.showerror(title='wrong',message='验证码应为4位')
        else:
            messagebox.showinfo(title='successful', message='登录成功')
            msg = '用户 %s 登录成功' % name.get()
            s.send(msg.encode('utf-8'))
            print("send len is : [%d]" % len(msg))
            menu()


class model:
    def __init__(self, name="test", dataset_name="squadv2", batch_size=32, epoch=4, lr="3e-5", output_dir="/tmp/"):
        self.name = name
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.epoch = epoch
        self.lr = lr
        self.output_dir = output_dir

    def print_cmd(self):
        return self.name + "?" + self.dataset_name + "?" + str(self.batch_size) + "?" + str(
            self.epoch) + "?" + self.lr + "?" + self.output_dir


def isInExcel(data):
    filename = "user_info.xls"
    excel = xlrd.open_workbook(filename, encoding_override="utf-8")
    sheet = excel.sheets()[0]

    sheet_row_mount = sheet.nrows  # 行数
    sheet_col_mount = sheet.ncols  # 列数

    sheet_name = []
    # 所有用户名信息
    for x in range(1, sheet_row_mount):
        y = 0
        sheet_name.append(sheet.cell_value(x, y))

    for x in sheet_name:
        # 找到用户名
        if (data == x):
            # 已有用户
            return 1
            break;
    # 未注册用户
    return -1


# 用户存在时，判断密码是否正确
def isPasswordDirect(data, passWord):
    filename = "user_info.xls"
    excel = xlrd.open_workbook(filename, encoding_override="utf-8")
    sheet = excel.sheets()[0]

    sheet_row_mount = sheet.nrows  # 行数
    sheet_col_mount = sheet.ncols  # 列数

    sheet_name = []
    # 所有用户名信息
    for x in range(1, sheet_row_mount):
        y = 0
        sheet_name.append(sheet.cell_value(x, y))
    sheet_passWord = []
    # 所有密码信息
    for x in range(1, sheet_row_mount):
        y = 1
        sheet_passWord.append(sheet.cell_value(x, y))

    for i in range(len(sheet_name)):
        if (data == sheet_name[i]):
            # 记录用户名在数组中的位置
            record = i
            break

    for i in range(len(sheet_passWord)):
        if (passWord == sheet_passWord[i]):
            # 判断用户名位置与密码位置是否相同
            # 以及密码是否与用户信息中的密码一致
            if (i == record):
                # 密码正确
                return 1
                break
    # 密码错误
    return -1



# 设计用户注册GUI界面
def registereds():
    registered = Tk()
    registered.title('registered')
    registered.geometry('230x185')
    Label(registered, text='用户注册').grid(row=0, column=0, columnspan=2)
    Label(registered, text='用户名：').grid(row=1, column=0, sticky=E)
    names = Entry(registered)
    names.grid(row=1, column=1)
    Label(registered, text='密码：').grid(row=2, column=0, sticky=E)
    passwds = Entry(registered, show='●')
    passwds.grid(row=2, column=1)
    Label(registered, text='确认密码：').grid(row=3, column=0)
    repasswd = Entry(registered, show='●')
    repasswd.grid(row=3, column=1)
    Label(registered, text='手机号：').grid(row=4, column=0, sticky=E)
    phonenum = Entry(registered)
    phonenum.grid(row=4, column=1)
    Label(registered, text='身份证号：').grid(row=5, column=0)
    man = Entry(registered)
    man.grid(row=5, column=1)

    # 判断是否含有特殊符号
    def teshufuhao(input_psd):
        string = "~!@#$%^&*()_+-*/<>,.[]\/?"
        for i in string:
            if i in input_psd:
                return True
        return False

    def registeredes():
        # 密码长度小于8
        if len(passwds.get()) < 8:
            messagebox.showerror(title='wrong', message='注册失败，密码不应少于8位')
        # 密码不同时含有数字、字母和特殊符号
        elif not (
                any([x.isdigit() for x in passwds.get()]) and any([x.isalpha() for x in passwds.get()]) and teshufuhao(
            passwds.get())):
            messagebox.showerror(title='wrong', message='注册失败，密码格式错误，必须包括字母和数字以及特殊符号')
        # 两次密码输入不一样
        elif passwds.get() != repasswd.get():
            messagebox.showerror(title='wrong', message='注册失败，两次密码不相同')
        # 手机号不正确
        elif not (phonenum.get().isdigit() and len(phonenum.get()) == 11):
            messagebox.showerror(title='wrong', message='注册失败，请输入正确的11位手机号')
        # 身份证号不正确
        elif len(man.get()) != 18:
            messagebox.showerror(title='wrong', message='注册失败，请输入正确的18位身份证号')
        else:
            messagebox.showinfo(title='successful', message='注册成功！')
            # 将新用户信息存入用户信息文件
            excel = xlrd.open_workbook('user_info.xls')
            sheet = excel.sheets()[0]

            nrow = sheet.nrows  # 文件行数

            wb = copy(excel)
            w_sheet = wb.get_sheet(0)
            # 从数据下一行开始写入新用户信息
            w_sheet.write(nrow, 0, names.get())
            w_sheet.write(nrow, 1, repasswd.get())
            w_sheet.write(nrow, 2, phonenum.get())
            w_sheet.write(nrow, 3, man.get())

            wb.save('user_info.xls')

    Button(registered, text='注册', command=registeredes).grid(row=6, column=0, columnspan=3)


def sys_exit():
    print("程序结束")
    msg = 'disconnect'
    # print("total send times is : %d " % receive_count)
    s.send(msg.encode('utf-8'))
    s.close()
    sys.exit("0")


def show_ans1():
    ans1 = Message(eval_page, text=data_ans.question_answer_id[0][1], font=("Times New Roman", 13), bg="AliceBlue",
                   width=400)
    ans1.place(x=50, y=360)


def show_ans2():
    ans1 = Message(eval_page, text=data_ans.question_answer_id[1][1], font=("Times New Roman", 13), bg="AliceBlue",
                   width=400)
    ans1.place(x=50, y=460)


def show_ans3():
    ans1 = Message(eval_page, text=data_ans.question_answer_id[2][1], font=("Times New Roman", 13), bg="AliceBlue",
                   width=400)
    ans1.place(x=50, y=560)


def check_pred1():
    id = data_ans.question_answer_id[0][-1]
    # [0]是预测值最高的
    res = data_pred[id]
    pred1 = Message(eval_page, text=res, font=("Times New Roman", 13), bg="CornflowerBlue",
                    width=400)
    pred1.place(x=50, y=330)


def check_pred2():
    id = data_ans.question_answer_id[1][-1]
    # [0]是预测值最高的
    res = data_pred[id]
    pred2 = Message(eval_page, text=res, font=("Times New Roman", 13), bg="CornflowerBlue",
                    width=400)
    pred2.place(x=50, y=430)


def check_pred3():
    id = data_ans.question_answer_id[2][-1]
    # [0]是预测值最高的
    res = data_pred[id]
    pred3 = Message(eval_page, text=res, font=("Times New Roman", 13), bg="CornflowerBlue",
                    width=400)
    pred3.place(x=50, y=530)


def eval_mode():
    msg = '进入测试模式'
    s.send(msg.encode('utf-8'))
    print("send len is : [%d]" % len(msg))
    # data_pred
    main_gui.withdraw()
    global eval_page
    eval_page = Tk()
    eval_page.geometry('600x600')
    eval_page.title('SQuAD2.0问答系统--测试模式')
    title = Label(eval_page, text="测试信息", font=("楷体", 20))
    title.place(x=240, y=10)

    context = Message(eval_page, text=data_ans.context, font=("Times New Roman", 13), bg="lightgreen", width=500)
    context.place(x=50, y=50)

    question1 = Message(eval_page, text=data_ans.question_answer_id[0][0], font=("Times New Roman", 13),
                        bg="PaleVioletRed", width=400)
    question1.place(x=50, y=300)
    pred1_bt = Button(eval_page, text='预测', command=check_pred1)
    pred1_bt.place(x=470, y=300)
    question1_bt = Button(eval_page, text='标答', command=show_ans1)
    question1_bt.place(x=520, y=300)
    question2 = Message(eval_page, text=data_ans.question_answer_id[1][0], font=("Times New Roman", 13),
                        bg="PaleVioletRed", width=400)
    question2.place(x=50, y=400)
    pred2_bt = Button(eval_page, text='预测', command=check_pred2)
    pred2_bt.place(x=470, y=400)
    question2_bt = Button(eval_page, text='标答', command=show_ans2)
    question2_bt.place(x=520, y=400)
    question3 = Message(eval_page, text=data_ans.question_answer_id[2][0], font=("Times New Roman", 13),
                        bg="PaleVioletRed", width=400)
    question3.place(x=50, y=500)
    pred3_bt = Button(eval_page, text='预测', command=check_pred3)
    pred3_bt.place(x=470, y=500)
    question3_bt = Button(eval_page, text='标答', command=show_ans3)
    question3_bt.place(x=520, y=500)

    exit_bt = Button(eval_page, text='退出', command=sys_exit)
    exit_bt.place(x=550, y=550)

    eval_page.mainloop()


def train_deploy():
    msg = '进入训练模式'
    s.send(msg.encode('utf-8'))
    print("send len is : [%d]" % len(msg))
    main_gui.withdraw()
    global train_para
    train_para = Tk()
    train_para.geometry('260x260')
    train_para.title('SQuAD2.0问答系统--训练部署')
    Label(train_para, text="模型参数", font=("楷体", 20)).grid(row=0, column=0, columnspan=2)
    Label(train_para, text="模型名称：", font=("楷体", 14)).grid(row=1, column=0)
    name = Entry(train_para)
    name.grid(row=1, column=1)
    Label(train_para, text="数据集： ", font=("楷体", 14)).grid(row=2, column=0)
    dataset_name = Entry(train_para)
    dataset_name.grid(row=2, column=1)
    Label(train_para, text="batch size: ", font=("Times New Roman", 14)).grid(row=3, column=0)
    batch_size = Entry(train_para)
    batch_size.grid(row=3, column=1)
    Label(train_para, text="epoch: ", font=("Times New Roman", 14)).grid(row=4, column=0)
    epoch = Entry(train_para)
    epoch.grid(row=4, column=1)
    Label(train_para, text="lr: ", font=("Times New Roman", 14)).grid(row=5, column=0)
    lr = Entry(train_para)
    lr.grid(row=5, column=1)
    Label(train_para, text="output dir: ", font=("Times New Roman", 14)).grid(row=6, column=0)
    output_dir = Entry(train_para)
    output_dir.grid(row=6, column=1)

    def check_input():
        server_gui.append(model=model(name=name.get(),
                                      dataset_name=dataset_name.get(),
                                      batch_size=int(batch_size.get()),
                                      epoch=int(epoch.get()),
                                      lr=lr.get(),
                                      output_dir=output_dir.get()))
        msg = server_gui.print_cmd()
        s.send(msg.encode('utf-8'))
        print("send len is : [%d]" % len(msg))
        trainandshow()

    train_bt = Button(train_para, text='确认', command=check_input)
    train_bt.place(x=150, y=220)
    exit_bt = Button(train_para, text='退出', command=sys_exit)
    exit_bt.place(x=220, y=220)
    train_para.mainloop()


def trainandshow():
    msg = '展示训练结果'
    s.send(msg.encode('utf-8'))
    print("send len is : [%d]" % len(msg))
    messagebox.showinfo(title='开始训练', message='开始训练')
    train_para.withdraw()
    global train_show
    train_show = Tk()
    train_show.geometry('400x450')
    train_show.title('SQuAD2.0问答系统--训练结果')
    title = Label(train_show, text="训练结果测评", font=("楷体", 20))
    title.place(x=130, y=50)
    data_use = data_process.data_process()

    model_para = server_gui.find_model()
    item = Label(train_show, text=("模型名称: " + model_para.name), font=("楷体", 12), bg="LightGreen")
    item.place(x=100, y=100)
    item1 = Label(train_show, text=("存储路径: " + model_para.output_dir), font=("楷体", 12), bg="LightGreen")
    item1.place(x=100, y=120)
    item2 = Label(train_show, text=("数据集名称： {num}".format(num=model_para.dataset_name)), font=("楷体", 12),
                  bg="LightGreen")
    item2.place(x=100, y=140)
    item3 = Label(train_show, text=("batch_size： {num}".format(num=model_para.batch_size)), font=("楷体", 12),
                  bg="LightGreen")
    item3.place(x=100, y=160)
    item3 = Label(train_show, text=("epoch： {num}".format(num=model_para.epoch)), font=("楷体", 12), bg="LightGreen")
    item3.place(x=100, y=180)
    item4 = Label(train_show, text=("lr： {num}".format(num=model_para.lr)), font=("楷体", 12), bg="LightGreen")
    item4.place(x=100, y=200)

    # 实验结果
    result = data_use.read_result()
    res1 = Label(train_show, text=("EM： {num}".format(num=result["exact"])), font=("楷体", 12), bg="Wheat")
    res1.place(x=100, y=250)
    res2 = Label(train_show, text=("F1 score： {num}".format(num=result["f1"])), font=("楷体", 12), bg="Wheat")
    res2.place(x=100, y=270)
    res3 = Label(train_show, text=("HasAns_exact： {num}".format(num=result["HasAns_exact"])), font=("楷体", 12),
                 bg="Wheat")
    res3.place(x=100, y=290)
    res4 = Label(train_show, text=("HasAns_f1： {num}".format(num=result["HasAns_f1"])), font=("楷体", 12),
                 bg="Wheat")
    res4.place(x=100, y=310)
    res5 = Label(train_show, text=("NoAns_exact： {num}".format(num=result["NoAns_exact"])), font=("楷体", 12),
                 bg="Wheat")
    res5.place(x=100, y=330)
    res6 = Label(train_show, text=("NoAns_f1： {num}".format(num=result["NoAns_f1"])), font=("楷体", 12),
                 bg="Wheat")
    res6.place(x=100, y=350)

    exit_bt = Button(train_show, text='退出', command=sys_exit)
    exit_bt.place(x=350, y=400)

    train_show.mainloop()


def display():
    main_gui.withdraw()
    global data_info_display
    data_info_display = Tk()
    # 设计窗口大小
    data_info_display.geometry('600x400')
    data_info_display.title('SQuAD2.0问答系统--后台数据集信息展示')
    title = Label(data_info_display, text="后台数据集信息展示", font=("楷体", 20))
    title.place(x=130, y=100)
    exit_bt = Button(data_info_display, text='退出', command=sys_exit)
    exit_bt.place(x=550, y=350)

    item = Label(data_info_display, text=("数据集名称: " + info.filename), font=("楷体", 12))
    item.place(x=100, y=170)
    item1 = Label(data_info_display, text=("数据集路径: " + info.path), font=("楷体", 12))
    item1.place(x=100, y=220)
    item2 = Label(data_info_display, text=("数据集包含了 {num} 个元素".format(num=info.dataset_num)), font=("楷体", 12))
    item2.place(x=100, y=270)
    # 插入信息
    data_info_display.mainloop()


def menu():
    login.withdraw()
    global main_gui
    main_gui = Tk()
    # info,data = read_json()
    main_gui.title('SQuAD2.0问答系统--主界面')
    # 设计窗口大小
    main_gui.geometry('600x400')
    title = Label(main_gui, text="SQuAD2.0问答系统--主界面", font=("楷体", 20))
    title.place(x=130, y=100)
    # Label(main_gui, text="退出").grid(row=0, column=0, columnspan=2)
    bt1 = Button(main_gui, text='训练模式', command=train_deploy, height=5, width=15)
    bt1.place(x=50, y=190)
    bt2 = Button(main_gui, text='测试模式', command=eval_mode, height=5,
                 width=15)
    bt2.place(x=250, y=190)
    bt3 = Button(main_gui, text='查看数据集信息', command=display, height=5,
                 width=15)
    bt3.place(x=450, y=190)
    exit_bt = Button(main_gui, text='退出', command=sys_exit)
    exit_bt.place(x=550, y=350)
    # Label(main_gui, text=info.dataset_num).grid(row=1, column=0)
    main_gui.mainloop()
    print("exit")
    login.destroy()


if __name__ == "__main__":
    # 启动客户端
    global s, info, data_ans, data_pred
    ip = '127.0.0.1'
    # ip = "10.1.114.125"
    port = 6000
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    failed_count = 0
    while True:
        try:
            print("start connect to server ")
            s.connect((ip, port))
            break
        except socket.error:
            failed_count += 1
            print("fail to connect to server %d times" % failed_count)
            if failed_count == 100:
                sys.exit(-11)

    read_data = data_process.data_process()
    info, data_ans, data_pred = read_data.read_json()
    login = Tk()
    # main_gui = Tk()

    login.title('登录界面')
    # 设计窗口大小
    login.geometry('210x200')

    # 设计GUI用户登录窗体
    Label(login, text="用户登录").grid(row=0, column=0, columnspan=2)
    Label(login, text="用户名").grid(row=1, column=0)
    name = Entry(login)
    name.grid(row=1, column=1)
    Label(login, text="密码：").grid(row=2, column=0)
    passWord = Entry(login, show='●')
    passWord.grid(row=2, column=1)
    user = User(name=name, passWord=passWord)
    # Button(login, text='发送验证码').grid(row=4, column=0, columnspan=3)
    Button(login, text='登录', command=user.successful).grid(row=5, column=0, columnspan=3)
    Button(login, text='注册', command=user.registereds).grid(row=6, column=0, columnspan=3)
    exit_bt = Button(login, text='退出', command=sys_exit)
    exit_bt.place(x=170, y=160)
    login.mainloop()
    s.close()

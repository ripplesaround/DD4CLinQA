# coding=utf-8

'''
Author: ripples
Email: ripplesaround@sina.com

date: 2021/5/28 22:15
desc:
'''

import socket
import sys
import os
import struct

SEND_BUF_SIZE = 256

RECV_BUF_SIZE = 256

Communication_Count: int = 0

receive_count: int = 0


def start_tcp_server(ip, port):
    # create socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (ip, port)

    # bind port
    print("starting listen on ip %s, port %s" % server_address)
    sock.bind(server_address)

    # get the old receive and send buffer size
    s_send_buffer_size = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
    s_recv_buffer_size = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
    print("socket send buffer size[old] is %d" % s_send_buffer_size)
    print("socket receive buffer size[old] is %d" % s_recv_buffer_size)

    # set a new buffer size
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, SEND_BUF_SIZE)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, RECV_BUF_SIZE)

    # get the new buffer size
    s_send_buffer_size = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
    s_recv_buffer_size = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
    print("socket send buffer size[new] is %d" % s_send_buffer_size)
    print("socket receive buffer size[new] is %d" % s_recv_buffer_size)

    # start listening, allow only one connection
    try:
        sock.listen(1)
    except socket.error:
        print("fail to listen on port %s")
        sys.exit(1)
    while True:
        print("waiting for connection")
        client, addr = sock.accept()
        print("having a connection")
        break
    msg = 'welcome to tcp server' + "\r\n"
    receive_count = 0
    receive_count += 1
    while True:
        print("\r\n")
        msg = client.recv(16384)
        msg_de = msg.decode('utf-8')
        print("recv len is : [%d]" % len(msg_de))
        print("###############################")
        print(msg_de)
        print("###############################")

        if msg_de == 'disconnect':
            break

        train_cmd = msg_de.split(" ")
        if train_cmd[0] == "train_model":
            print("开始训练")
            train_cmd = train_cmd[-1].split("?")
            python_path = "/home/fwx/anaconda3/envs/py37/bin/python3.7 "
            train_file_path = "/home/fwx/project/dd_2020/SQuAD_v2/run_qa.py "
            dataset_name = "--dataset_name " + train_cmd[1]
            batch_size = " --per_device_train_batch_size "+train_cmd[2]
            epoch = " --num_train_epochs "+train_cmd[3]
            lr = " --learning_rate "+train_cmd[4]
            output_dir = " --output_dir " + train_cmd[5]
            max_seq_length_and_doc_stride = " --max_seq_length 384 --doc_stride 128"

            basic_para = "--model_name_or_path bert-base-uncased " + dataset_name + " --do_train --do_eval --version_2_with_negative"
            basic_para += (lr+epoch+max_seq_length_and_doc_stride+output_dir+batch_size)
            train_cmd = python_path + train_file_path + basic_para
            print(train_cmd)

            os.system(train_cmd)
            # 参数实例
            # --model_name_or_path
            # bert - base - uncased
            # --dataset_name
            # squad_v2
            # --do_train
            # --do_eval
            # --version_2_with_negative
            # --learning_rate
            # 3e-5
            # --num_train_epochs
            # 4
            # --max_seq_length
            # 384
            # --doc_stride
            # 128
            # --output_dir
            # / home / fwx / tmp / squadv2 / exp_org /
            # --per_device_train_batch_size
            # 12
            #
            # /home/fwx/anaconda3/envs/py37/bin/python3.7 -u /home/fwx/project/dd_2020/SQuAD2.0_code/run_squad_org.py --model_type bert --model_name_or_path bert-large-uncased --do_train --do_eval --do_lower_case --version_2_with_negative --train_file /home/fwx/project/dd_2020/SQUAD_DATASET/squad2.0/train-v2.0.json --predict_file /home/fwx/project/dd_2020/SQUAD_DATASET/squad2.0/dev-v2.0.json --per_gpu_train_batch_size=12 --per_gpu_eval_batch_size=2 --learning_rate 3e-5 --num_train_epochs 4.0 --max_seq_length 384 --doc_stride 128 --save_steps 6000 --output_dir /home/fwx/tmp/squad_baseline_final_large_2.0/ --overwrite_output_dir

        msg = ("hello, client, i got your msg %d times, now i will send back to you " % receive_count)
        client.send(msg.encode('utf-8'))
        receive_count += 1
        print("send len is : [%d]" % len(msg))

    print("finish test, close connect")
    client.close()
    sock.close()
    print(" close client connect ")


if __name__ == '__main__':
    start_tcp_server('127.0.0.1', 6000)

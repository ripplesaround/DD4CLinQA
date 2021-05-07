# coding=utf-8

'''
Author: ripples
Email: ripplesaround@sina.com

date: 2021/5/6 11:56
desc: 自动生成实验报告
'''
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

class AutoLabReport:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.story =[]
        pdfmetrics.registerFont(TTFont('msyh', '/Users/ginne/Library/Fonts/YAHEI.ttf'))  ### 设置中文字体名称为msyh


    def get_text(self,text,style=None):
        """
        生成对应的样式，塞入实验报告中
        :param text:
        :param style:
        :return:
        """
        if style is None:
            style = self.styles['Normal']
        self.story.append(Paragraph(text, style))


    def build_pdf(self,filename=None):
        if filename is None:
            filename = "hello.pdf"
        doc = SimpleDocTemplate(filename)
        doc.build(self.story)
        print("***成功生成 {name}***".format(name=filename))
